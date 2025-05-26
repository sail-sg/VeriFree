# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from dataclasses import dataclass, field
from typing import List, Literal
from typing import List, Optional, Tuple
from tqdm import tqdm
import gc
import numpy as np
import tree

import launchpad as lp
from jinja2 import Template
import torch
from torch.utils.data import DataLoader
import vllm
import torch.distributed as dist

from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import masked_mean, masked_whiten
from collections import defaultdict
from oat.utils.ipc import PlasmaShmClient

from utils.types import VeriFreeTrajectoryData
from utils.data import VeriFreeTrajectoryDataset
from utils.collector import FeedbackCollector 


def apply_ours_qwen3_template(question: str):
    prompt_template_jinja = """\
<|im_start|>user\n{{question}}\nPlease reason step by step, and put your final answer within <answer> \\boxed{} </answer>.<|im_end|>\n<|im_start|>assistant\n
"""
    prompt_instruction_template = Template(prompt_template_jinja)
    prompt_instruction = prompt_instruction_template.render(question=question)
    return prompt_instruction


TEMPLATE_FACTORY = {
    "ours": apply_ours_qwen3_template,
}

@dataclass
class VeriFreeArgs(PPOArgs):
    # Template.
    prompt_template: Literal["ours",] = field(default="ours")
    # Scheduling.
    increase_max_tokens_at: int = np.inf
    new_max_tokens: int = 5000
    vanilla_adv: bool = field(default=True)
    critic_type: Literal["grpo", "drgrpo", "rloo", "naive"] = "rloo"
    reward_source: str = field(
        default="logp",
        metadata={"help": "Choose the p or logp as reward."},
    )
    sft_coef_source: str = field(
        default="adv",
        metadata={"help": "Choose the adv or reward or 1 as reward."},
    )


class VeriFreeActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        
        self.step_counter = 0
        
        # Special treatment to sample from a base model - now only cover Qwen.
        if "qwen" in args.pretrain.lower():
            if args.prompt_template in ['ours']:
                
                self.sampling_params = vllm.SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.generate_max_length,
                    stop = ["</answer>"],
                    n=args.num_samples,
                    logprobs=2,
                    include_stop_str_in_output=True,
                    skip_special_tokens=False,
                )
                self.eval_sampling_params = vllm.SamplingParams(
                    n=args.eval_n,
                    temperature=args.eval_temperature,
                    top_p=args.eval_top_p,
                    top_k=args.eval_top_k,
                    max_tokens=args.eval_generate_max_length,
                    stop = ["</answer>"],
                    include_stop_str_in_output=True,
                    skip_special_tokens=False,
                )
            else:
                raise ValueError(f"unrecognized prompt_template: {args.prompt_template}")
        else:
            raise ValueError("There may be model-template mismatch.")
        
    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[VeriFreeTrajectoryData]:
        # Schedule the sampling budget.
        self.step_counter += 1
        if self.step_counter == self.args.increase_max_tokens_at:
            self.sampling_params.max_tokens = self.args.new_max_tokens
            self.eval_sampling_params.max_tokens = self.args.new_max_tokens
            self.reset_prefix_cache()
        
        assert not self.eval_mode
        info = {}
        logging.info(f"actor start")

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        think_ids = []
        answer_ids = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids # (tuple)
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]

                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))
                
        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        st = time.time()
        
        end_with_eos = []
        for i in range(len(outputs)):
            think_ids.append([])
            end_with_eos.append([])
            answer_ids.append([])
            for k in range(self.sampling_params.n):
                model_response = outputs[i].outputs[k].text
                
                if self.args.prompt_template == 'ours':
                    think_index = model_response.find("<answer")
                    if think_index != -1:
                        think_text = model_response[:think_index] + "<answer"
                        think_id = tuple(self.tokenizer(think_text)['input_ids'])
                        answer_text = Template("> \\boxed{ {{-reference-}} } </answer>").render(reference=references[i])
                        answer_id = tuple(self.tokenizer(answer_text)['input_ids'])
                        if len(think_id + answer_id) > self.sampling_params.max_tokens:
                            no_eos[i*self.sampling_params.n + k] = True
                        end_with_eos[i].append(False)
                    else:
                        think_text = model_response
                        think_id = outputs[i].outputs[k].token_ids
                        answer_text = ""
                        answer_id = tuple()
                        end_with_eos[i].append(True)
                    think_ids[i].append(think_id)
                    answer_ids[i].append(answer_id)
                else:
                    raise NotImplementedError(f'Not supported template: {self.args.prompt_template}')
                
        no_eos = np.array(no_eos).reshape(len(prompts), -1)

        info["actor/verify_time"] = time.time() - st

        info["actor/no_eos_count"] = no_eos.sum()
        info["actor/num_data"] = no_eos.size
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature
        info["actor/end_with_eos"] = np.mean(end_with_eos)

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                trajectory_data.append(
                    VeriFreeTrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        gt_text=self.tokenizer.decode(list(prompt_token_ids[i]) + list(think_ids[i][j]) + list(answer_ids[i][j])),
                        think_text=self.tokenizer.decode( list(think_ids[i][j]) ),
                        answer_text=self.tokenizer.decode( list(answer_ids[i][j]) ),
                        think_ids=think_ids[i][j],
                        answer_ids=answer_ids[i][j],
                        loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                        no_eos=no_eos[i][j],
                        end_with_eos=end_with_eos[i][j],
                        info=info,
                    )
                )
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle
    



class VeriFreeLearner(PPOLearner):
    def _init(self, args: VeriFreeArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.args = args
        self.dataset_builder = VeriFreeTrajectoryDataset
        
        self.collector = FeedbackCollector(
            args, actors, PlasmaShmClient(self.ipc_server)
        )

    def _apply_template(self, example):
        problem = example[self.args.input_key]
        example[self.args.input_key] = TEMPLATE_FACTORY[args.prompt_template](problem)
        return example

    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[args.train_split].select(
            range(min(args.max_train, len(prompt_dataset[args.train_split])))
        )

        prompts_data = prompts_data.map(lambda x: self._apply_template(x))

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=args.input_key,
            output_key=args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def eval_dataloader_collate_fn(self, item_list):
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[args.prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers

    def evaluate(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        for benchmark_name, dataset in self.eval_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )
        return all_metrics
    
    
    def learn(self, learning_round: int):
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size_per_device, 
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
        )
        self.model.eval()
        device = torch.cuda.current_device()
        collected_logps = []
        for batch in dataloader:
            completion_masks = self.get_completion_mask(batch['gt_attention_mask'], batch['prompt_ids_lens'])
            response_masks = completion_masks[:, 1:] 
            
            batch_logits = self.model(
                batch['gt_input_ids'].to(device), attention_mask=batch['gt_attention_mask'].to(device)
            )["logits"].detach().float()
            batch_logits /= self.args.temperature

            batch_logps = self.get_batch_logps(
                batch_logits,
                batch['gt_input_ids'].to(device),
                response_masks.to(device),
            )
            batch_logps = batch_logps.cpu()
            
            
            batch_logps_list = [batch_logps[bid] for bid in range(len(batch_logps))]
            del batch_logits 
            del batch_logps
            collected_logps.extend(batch_logps_list)
            
        assert len(collected_logps) == len(dataset.trajectories)
        for traj_idx, (logps, traj) in enumerate(zip(collected_logps, dataset.trajectories)):
            prompt_len = traj['prompt_ids_lens']
            think_len = traj['think_ids_lens']
            answer_len = traj['answer_ids_lens']

            if answer_len == 0:
                logp_reward = torch.tensor(float('-inf'))
            else:
                logp_reward = logps[prompt_len + think_len - 1 : prompt_len + think_len + answer_len - 1].sum() 
                
            traj["gt_action_logprobs"] = logps[prompt_len - 1 : prompt_len + think_len + answer_len - 1].detach()
            traj["logp_reward"] = logp_reward.detach()
            if traj["no_eos"]:
                traj["logp_reward"] = torch.tensor(float('-inf'))
            self.pi_buffer[traj_idx].gt_action_logprobs = traj["gt_action_logprobs"].tolist()
            self.pi_buffer[traj_idx].logp_reward = traj["logp_reward"].item()

        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])

        # Load all buffered data, and PPO will iterate through inner loops.
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        if self.critic is not None:
            self.critic.train()
        st = time.time()

        logging.info(
            f"start learn() buffer_len={len(self.pi_buffer)} dl_len={len(dataloader)}"
        )
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.grad_acc_step
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.grad_acc_step == 0:

                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        logging.info(f"finish learn()")

        return train_info

    def compute_ppo_advantages(self, rewards, input_ids, att_mask, response_masks):
        all_values = []

        with torch.no_grad():
            for i in range(
                0, len(input_ids), self.args.train_batch_size_per_device
            ):
                batch_inds = torch.arange(
                    i, i + self.args.train_batch_size_per_device
                )
                ## Forward critic network.
                batch_values = self.critic(
                    input_ids=input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                )
                batch_value_masks = att_mask[batch_inds].clone()[:, 1:]
                batch_value_masks = torch.concat(
                    [
                        batch_value_masks,
                        torch.zeros(len(batch_value_masks), 1, device=att_mask.device),
                    ],
                    axis=1,
                )
                batch_values = (batch_values * batch_value_masks)[:, :-1]
                all_values.append(batch_values)
        values = torch.cat(all_values)

        # Compute gae (for policy learning) and return (for critic learning); vectorize later.
        advantages = torch.zeros_like(rewards)
        for i in range(len(advantages)):
            action_inds = torch.where(response_masks[i])[0]
            lastgaelam = 0
            for t in reversed(action_inds):
                nextvalues = values[i, t + 1] if t < action_inds[-1] else 0.0
                delta = rewards[i, t] + self.args.gamma * nextvalues - values[i, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages[i, t] = lastgaelam

        returns = advantages + values
        advantages = masked_whiten(advantages, response_masks)

        return advantages, returns, values

    def compute_grpo_advantages(self, rewards, response_masks):
        rewards = rewards.sum(-1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.args.num_samples).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.args.num_samples, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if not self.args.vanilla_adv:
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages
    
    def compute_rloo_advantages(self, rewards, response_masks):
        rewards = rewards.sum(-1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.args.num_samples).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.args.num_samples, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) * (self.args.num_samples / (self.args.num_samples - 1))
        
        return advantages

    def learning_step(self, trajectory):
        args: PPOArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["gt_input_ids"].to(device)
        att_mask = trajectory["gt_attention_mask"].to(device)
        final_rewards = (
            torch.tensor(trajectory["logp_reward"])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        ################  exp(logp) ################
        if args.reward_source == 'p':
            final_rewards = torch.exp(final_rewards)
        ############################################

        prompt_id_lens = trajectory["prompt_ids_lens"]
        
        assert final_rewards.shape[0] == input_ids.shape[0]
        action_logprobs = [
            torch.tensor(lp).to(device) for lp in trajectory["gt_action_logprobs"]
        ]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]
        think_masks, answer_masks = self.get_think_and_answer_mask(
            att_mask, 
            prompt_id_lens, 
            trajectory['think_ids_lens'], 
            trajectory['answer_ids_lens']
        )
        think_masks = think_masks[:, 1:]
        answer_masks = answer_masks[:, 1:]

        logging.info(f"learn data size {input_ids.shape}")

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        # Forward old models.
        ## 1) (Option 1) Policy log probabilities are directly from actors (vLLM).
        logps = torch.zeros_like(response_masks).float()
        for i in range(len(logps)):
            logps[i, torch.where(response_masks[i])[0]] = action_logprobs[i]
        ## 2) (Option 2) Reevaluate log probabilities using learner model.
        # all_logps = []
        # with torch.no_grad():
        #     for i in range(0, len(input_ids), args.mini_train_batch_size_per_device):
        #         batch_inds = torch.arange(i, i + args.mini_train_batch_size_per_device)

        #         batch_logits = self.model(
        #             input_ids[batch_inds], attention_mask=att_mask[batch_inds]
        #         )["logits"].float()
        #         batch_logits /= args.temperature
        #         batch_logps = self.get_batch_logps(
        #             batch_logits,
        #             input_ids[batch_inds],
        #             response_masks[batch_inds],
        #         )
        #         all_logps.append(batch_logps)
        #     logps = torch.cat(all_logps)
        #     del all_logps

        ## 2) Reference.
        if self.ref_model is not None:
            all_ref_logps = []
            with torch.no_grad():
                for i in range(
                    0, len(input_ids), args.train_batch_size_per_device
                ):
                    batch_inds = torch.arange(
                        i, i + args.train_batch_size_per_device
                    )

                    batch_ref_logits = self.ref_model(
                        input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                    )["logits"].float()
                    batch_ref_logits /= args.temperature
                    batch_ref_logps = self.get_batch_logps(
                        batch_ref_logits,
                        input_ids[batch_inds],
                        response_masks[batch_inds],
                    )
                    all_ref_logps.append(batch_ref_logps)
            ref_logps = torch.cat(all_ref_logps)

            # Combine final reward and kl penalty as rewards.
            kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            rewards = kl_rewards.clone()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "grpo":
            advantages = self.compute_grpo_advantages(rewards, response_masks)[:, None]
        elif self.args.critic_type == "rloo":
            advantages = self.compute_rloo_advantages(rewards, response_masks)
            advantages = advantages[:, None]
        elif self.args.critic_type == "naive":
            advantages = rewards.sum(dim=-1)
            
        # Compute losses and update models for multiple PPO epochs.
        stats = defaultdict(list)
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.train_batch_size_per_device):
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_final_rewards = final_rewards[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_think_masks = think_masks[mini_batch_inds]
                mb_answer_masks = answer_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                # # Further reduce valid token num to speed up IF:
                # ## 1. We only have PG loss, i.e., args.beta == 0.
                # ## 2. Advantage is zero in bandit case (e.g., GRPO).
                # ## 3. mini_train_batch_size_per_device is 1.
                # if (
                #     args.beta == 0
                #     and self.args.critic_type == "grpo"
                #     and len(mb_advantage) == 1
                # ):
                #     zero_adv = (mb_advantage == 0).item()  # bool
                #     if zero_adv:
                #         mb_last_valid_token_pos = 7  # An unimportant magic number.
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_think_masks = mb_think_masks[:, : mb_last_valid_token_pos - 1]
                mb_answer_masks = mb_answer_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                # Policy learning.
                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                logits /= args.temperature
                new_logps = self.get_batch_logps(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                
                ################## pg #####################################
                if args.reinforce_update:
                    pg_loss = -mb_advantage * new_logps
                else:
                    logprobs_diff = new_logps - mb_logps
                    ratio = torch.exp(logprobs_diff)
                    pg_loss = -mb_advantage.detach() * (new_logps * mb_think_masks).sum(dim=1) 
                    stats["ratio_max"].append(ratio.detach().max().item())
                    stats["ratio_min"].append(ratio.detach().min().item())
                
                pg_loss = (pg_loss * mb_loss_masks  ).mean()
                infos["pg_loss"] = pg_loss.detach()
                ################## pg #####################################
                
                
                ################## sft #####################################
                if args.sft_coef_source == 'adv':
                    sft_loss = (-new_logps * mb_answer_masks * mb_advantage.detach()).sum(dim=1)
                elif args.sft_coef_source == 'reward':
                    sft_loss = (-new_logps * mb_answer_masks * mb_final_rewards.detach() ).sum(dim=1) 
                elif args.sft_coef_source == '1':
                    sft_loss = (-new_logps * mb_answer_masks).sum(dim=1)
                else:
                    raise ValueError(f'Unrecognized sft_coef_source: {args.sft_coef_source}')

                sft_loss = (sft_loss * mb_loss_masks  ).mean()
                infos["sft_loss"] = sft_loss.detach()
                ################## sft #####################################
                
                ############ merge ############
                # num_response_tokens = mb_response_masks.sum()
                loss = (pg_loss + sft_loss) #/ num_response_tokens
                ############ merge ############
                
                infos["loss"] = loss.detach()
                if args.beta > 0:
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : mb_last_valid_token_pos - 1]
                    # k3 kl: http://joschu.net/blog/kl-approx.html.
                    # clamp to avoid numerical instability.
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio  # expm1 is more stable.
                    infos["kl3"] = (kl3 * mb_response_masks).detach().sum(1).mean()

                    reg_loss = masked_mean(kl3, mb_response_masks, axis=1)
                    reg_loss = args.beta * (reg_loss * mb_loss_masks).mean()
                    infos["reg_loss"] = reg_loss.detach()
                    loss += reg_loss

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)


        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        if not args.reinforce_update:
            infos["ratio_max"] = torch.tensor(stats["ratio_max"]).max()
            infos["ratio_max"] = torch.tensor(stats["ratio_max"]).max()
            infos["zero_pg_loss_count"] = (
                torch.tensor(stats["zero_pg_loss_count"]).float().mean()
            )
            infos["pg_clipfrac"] = torch.tensor(stats["pg_clipfrac"]).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        if args.reward_source == 'p':
            infos["p_reward_mean"] = final_rewards.mean().cpu()
            infos["p_reward_min"] = final_rewards.min().cpu()
            infos["p_reward_max"] = final_rewards.max().cpu()
        else:
            infos["logp_reward_mean"] = final_rewards.mean().cpu()
            infos["logp_reward_min"] = final_rewards.min().cpu()
            infos["logp_reward_max"] = final_rewards.max().cpu()
        infos["all_zero_rewards_count"] = (final_rewards.mean(-1) == 0).sum().cpu()
        infos["all_one_rewards_count"] = (final_rewards.mean(-1) == 1).sum().cpu()

        return infos


    def get_think_and_answer_mask(
        self,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
        think_id_lens: List[int],
        answer_id_lens: List[int]
    ):
        think_masks = attention_mask.clone().bool()
        answer_masks = attention_mask.clone().bool()
        # mask prompts
        for think_mask, answer_mask, prompt_len, think_len, answer_len in zip(think_masks, answer_masks, prompt_id_lens, think_id_lens, answer_id_lens):
            think_mask[:prompt_len] = False
            think_mask[prompt_len+think_len:] = False 
            answer_mask[:prompt_len+think_len] = False
            answer_mask[prompt_len+think_len+answer_len:] = False
        return think_masks, answer_masks



def run_verifree(args: VeriFreeArgs):
    learner_cls = VeriFreeLearner
    actor_cls = VeriFreeActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")

    args: VeriFreeArgs = get_default_args(VeriFreeArgs)

    args.algo = "PPO"
    args.online_evaluation = True  

    args = default_args_validation(args)
    run_verifree(args)
