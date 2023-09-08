from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--prompt",
        type=str,
        default='alpaca',
        help="wiz, alpaca",
    )
    parser.add_argument(
        "--model_name_tag",
        type=str,
        default='name',
        help="",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="")
    parser.add_argument("--top_p", type=float, default=1.0, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="../cache/")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir="../cache/")

    model.to(device)
    model.eval()

    if args.prompt == 'alpaca':
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
    elif args.prompt == 'wiz':
        prompt_no_input = "{instruction}\n\n### Response:"
    
    dataset_path = 'alpaca_eval/alpaca_eval_data.jsonl'
    prompt_key = 'instruction'

    with open(dataset_path) as f:
        results = []
        dataset = list(f)
        for point in tqdm(dataset):
            point = json.loads(point)
            instruction = point[prompt_key]
            prompt = prompt_no_input.format_map({"instruction":instruction})
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            generate_ids = model.generate(
                input_ids, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
                )
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            new_point = {}
            new_point["dataset"] = point["dataset"]
            new_point["instruction"] = point["instruction"]
            new_point["output"] = outputs.split("Response:")[1]
            new_point["generator"] = args.model_name_tag

            results.append(new_point)

    output_dir =  os.path.join(args.model_name_or_path, 'test_inference_alpaca_eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_name = "model_outputs.json"
    with open(os.path.join(output_dir, saved_name), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()