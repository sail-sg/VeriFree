"""
The evaluation script for the GPQA dataset.
Built upon the following repos:
General-Reasoner (https://github.com/TIGER-AI-Lab/General-Reasoner)
simple-evals (https://github.com/openai/simple-evals)
Thanks for their contributions to the community.
"""
import os
import re
import random
import pandas
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
from copy import deepcopy


QWEN3_GPQA_QUERY_TEMPLATE = """
<question>

A: <a>
B: <b>
C: <c>
D: <d>

Please reason step by step, and put your final answer within \\boxed{}.
Please only provide the letter of the answer in the box.
""".strip()


OURS_GPQA_QUERY_TEMPLATE = """
<question>

A: <a>
B: <b>
C: <c>
D: <d>

Please reason step by step, and put your final answer within <answer> \\boxed{} </answer>.
Please only provide the letter of the answer in the box.
""".strip()


def extract_last_boxed(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def get_prediction(output):
    return extract_last_boxed(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--n_repeats", type=int, required=True, help="number of repeats for each example")
    parser.add_argument("--disable_thinking", action="store_true", help="disable thinking mode")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=4,)
    
    if args.model_path.startswith("Qwen/Qwen3-"):
        GPQA_QUERY_TEMPLATE = QWEN3_GPQA_QUERY_TEMPLATE
    elif args.model_path.endswith("-VeriFree"):
        GPQA_QUERY_TEMPLATE = OURS_GPQA_QUERY_TEMPLATE
    else:
        raise ValueError("Unsupported model path.")
    
    variant = "diamond"
    df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
    examples = [row.to_dict() for _, row in df.iterrows()]
    rng = random.Random(0)
    new_examples = []
    for example in examples:
        for _ in range(args.n_repeats):
            example = deepcopy(example)
            permutation = rng.sample(range(4), 4)
            example["permutation"] = permutation
            new_examples.append(example)
    examples = new_examples
    
    success, fail = 0, 0
    answers = []
    all_token_length = []
    
    print('----------------- Start Answering -------------------')
    prompts = []
    correct_answers = []
    for row in examples:
        choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
        choices = [choices[i] for i in row["permutation"]]
        correct_index = choices.index(row["Correct Answer"])
        correct_answer = "ABCD"[correct_index]
        content = GPQA_QUERY_TEMPLATE.replace("<question>", row["Question"])
        content = content.replace("<a>", choices[0])
        content = content.replace("<b>", choices[1])
        content = content.replace("<c>", choices[2])
        content = content.replace("<d>", choices[3])
        
        messages = [{
            "role": "user",
            "content": content,
        }]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=not args.disable_thinking)
            
        prompts.append(prompt)
        correct_answers.append(correct_answer)
    
    ## Please choose proper sampling_params
    # sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=8192)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=8192)
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=8192)
    # sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=32768)
    # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768)
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=32768)
    outputs = llm.generate(prompts, sampling_params)
    
    assert len(correct_answers) == len(outputs), "Number of correct answers does not match number of outputs"
    
    for correct_answer, output in zip(correct_answers, outputs):
        answer = output.outputs[0].text
        all_token_length.append(len(output.outputs[0].token_ids))
        prediction = get_prediction(answer)
        if correct_answer == prediction:
            success += 1
        else:
            fail += 1

    print("\n----- Token Length Report -----")
    print(f'token length:  (mean) {np.mean(all_token_length):.2f}, (median) {np.median(all_token_length):.2f}')
    
    print("\n----- Accuracy Report -----")
    total_predictions = success + fail
    avg_acc = success / total_predictions if total_predictions > 0 else 0.0
    print(f"\Average Accuracy: {avg_acc*100:.2f}%")
    
    
    
    
