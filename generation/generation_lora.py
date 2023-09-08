from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict
import torch
import argparse
import json
import os
from tqdm import tqdm
import sys

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
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--training_data_source_name",
        type=str,
        default='alpaca',
        help="The training_data_source_name.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--base_model", type=str, default='jploski/llama-7b-hf')
    parser.add_argument('--lora_r', type=int, default=8, help='lora_r')
    parser.add_argument("--lora_target_modules", type=str, default='[q_proj,k_proj]')
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model, cache_dir="../cache/")
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="../cache/"
    )
    lora_target_modules = args.lora_target_modules.strip('[]').split(',')
    lora_target_modules = [s.strip() for s in lora_target_modules]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    checkpoint_name = os.path.join(args.model_name_or_path, "pytorch_model.bin")
    adapters_weights = torch.load(checkpoint_name,map_location="cuda:0")
    set_peft_model_state_dict(model, adapters_weights)

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if(args.training_data_source_name=='alpaca'or args.training_data_source_name=='alpaca_gpt4'):
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    
    if(args.dataset_name=="vicuna"):
        dataset_path = './test_data/vicuna_test_set.jsonl'
        prompt_key = 'text'
    elif(args.dataset_name=="koala"):
        dataset_path = './test_data/koala_test_set.jsonl'
        prompt_key = 'prompt'
    elif(args.dataset_name=="sinstruct"):
        dataset_path = './test_data/sinstruct_test_set.jsonl'
        prompt_key = 'instruction'
    elif(args.dataset_name=="wizardlm"):
        dataset_path = './test_data/wizardlm_test_set.jsonl'
        prompt_key = 'Instruction'
    elif(args.dataset_name=="truthfulqa"):
        dataset_path = './test_data/truthfulqa_test_set.jsonl'
        prompt_key = 'Question'
    elif(args.dataset_name=="lima"):
        dataset_path = './test_data/lima_test_set.jsonl'
        prompt_key = 'conversations'

    with open(dataset_path) as f:
        results = []
        dataset = list(f)
        for point in tqdm(dataset):
            point = json.loads(point)
            instruction = point[prompt_key]
            if(args.dataset_name=="sinstruct"):
                instances = point['instances']
                assert len(instances) == 1
                if  instances[0]['input']:
                    prompt = prompt_input.format_map({"instruction":instruction, 'input':instances[0]['input']})
                else:
                    prompt = prompt_no_input.format_map({"instruction":instruction})
            else:
                prompt = prompt_no_input.format_map({"instruction":instruction})
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            generate_ids = model.generate(input_ids=input_ids, max_new_tokens=args.max_length)
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            point['raw_output'] = outputs
            point['response'] = outputs.split("Response:")[1]
            results.append(point)

    output_dir =  os.path.join(args.model_name_or_path, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_name = args.dataset_name + "_" + str(args.seed) + '_' + str(args.max_length) + ".json"
    with open(os.path.join(output_dir, saved_name), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()