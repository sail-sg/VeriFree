# Copyright 2024 Garena Online Private Limited
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

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from oat.types import TrajectoryData
from oat.utils.deepspeed import DeepspeedStrategy
from oat.utils.data import zero_pad_sequences



class VeriFreeTrajectoryDataset(Dataset):
    def __init__(
        self,
        buffer: List[TrajectoryData],
        tokenizer: Callable,
        strategy: DeepspeedStrategy=None,
        **_,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        # Storing training data.
        self.trajectories = []
        
        if strategy is None:
            prograss_bar = range(len(buffer))
        else:
            prograss_bar = tqdm(
                range(len(buffer)),
                disable=not strategy.is_rank_0(),
                desc="Constructing ppo dataset",
            )

        for i in prograss_bar:
            trajectory_ids = list(buffer[i].prompt_ids) + list(buffer[i].response_ids)
            gt_trajectory_ids = list(buffer[i].prompt_ids) + list(buffer[i].think_ids) + list(buffer[i].answer_ids)
            traj_dict = {
                "input_ids": torch.tensor(trajectory_ids),
                "attention_mask": torch.ones(len(trajectory_ids)),
                "action_ids": buffer[i].response_ids,
                "gt_input_ids": torch.tensor(gt_trajectory_ids),
                "gt_attention_mask": torch.ones(len(gt_trajectory_ids)),
                "think_ids": buffer[i].think_ids,
                "answer_ids": buffer[i].answer_ids,
                "loss_mask": buffer[i].loss_mask,
                "prompt_ids_lens": len(buffer[i].prompt_ids),
                "think_ids_lens": len(buffer[i].think_ids),
                "answer_ids_lens": len(buffer[i].answer_ids),
                "action_logprobs": buffer[i].response_logprobs,
                "no_eos": buffer[i].no_eos,
                "end_with_eos": buffer[i].end_with_eos,
            }
            traj_dict["gt_action_logprobs"] = torch.ones_like(torch.tensor(traj_dict['action_logprobs'])) * 99999. # placeholder
            traj_dict["logp_reward"] = float('-inf')
            self.trajectories.append(traj_dict)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def collate_fn(self, item_list):
        batch_trajectories = {
            "input_ids": [],
            "action_ids": [],
            "attention_mask": [],
            "gt_input_ids": [],
            "gt_attention_mask": [],
            "think_ids": [],
            "answer_ids": [],
            "loss_masks": [],
            "prompt_ids_lens": [],
            "think_ids_lens": [],
            "answer_ids_lens": [],
            "action_logprobs": [],
            "gt_action_logprobs": [],
            "logp_reward": [],
            "end_with_eos": [],
        }
        for t in item_list:
            batch_trajectories["input_ids"].append(t["input_ids"])
            batch_trajectories["attention_mask"].append(t["attention_mask"])
            batch_trajectories["gt_input_ids"].append(t["gt_input_ids"])
            batch_trajectories["gt_attention_mask"].append(t["gt_attention_mask"])
            batch_trajectories["think_ids"].append(t["think_ids"])
            batch_trajectories["answer_ids"].append(t["answer_ids"])
            batch_trajectories["loss_masks"].append(t["loss_mask"])
            batch_trajectories["prompt_ids_lens"].append(t["prompt_ids_lens"])
            batch_trajectories["think_ids_lens"].append(t["think_ids_lens"])
            batch_trajectories["answer_ids_lens"].append(t["answer_ids_lens"])
            batch_trajectories["action_logprobs"].append(t["action_logprobs"])
            batch_trajectories["action_ids"].append(t["action_ids"])
            batch_trajectories["gt_action_logprobs"].append(t["gt_action_logprobs"])
            batch_trajectories["logp_reward"].append(t['logp_reward'])
            batch_trajectories["end_with_eos"].append(t['end_with_eos'])

        padding_side = "right"
        
        batch_trajectories["gt_input_ids"] = zero_pad_sequences(
            batch_trajectories["gt_input_ids"],
            side=padding_side,
            value=self.tokenizer.pad_token_id,
        )
        batch_trajectories["gt_attention_mask"] = zero_pad_sequences(
            batch_trajectories["gt_attention_mask"],
            side=padding_side,
        )
        return batch_trajectories
