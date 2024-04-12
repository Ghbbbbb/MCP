# run_llama.py

import argparse
import json
from models.llama import Llama
from utils import calculate_metric_and_output_file_llama
import os
import random

def main():
    default_config_file = "config.json"
    with open(default_config_file, 'r') as config_file:
        default_config = json.load(config_file)

    parser = argparse.ArgumentParser(description="Llama test")
    parser.add_argument("--ckpt_dir", type=str, default="models/CodeLlama-7b-Instruct", help="Path to checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str,default="models/CodeLlama-7b-Instruct/tokenizer.model", help="Path to tokenizer model")
    parser.add_argument("--max_seq_len", type=int, default=5000, help="Maximum sequence length")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size")
    parser.add_argument("--input_file", type=str, default="data/zh/HITD.json", help="Path to input JSON file")
    parser.add_argument("--prompt", type=str, default="prompt/zh/MCP_5shot_HITD.json", help="Path to prompt file")
    parser.add_argument("--output", type=str, default="outputs", help="Path to output file")

    args = parser.parse_args()
    model_name = os.path.basename(args.ckpt_dir)
    print("model_name:",model_name)
    if "robustness" in args.input_file:
        method = "MCP_5shot_" + os.path.basename(args.input_file)
    else:    
        method = os.path.basename(args.prompt)
    print("method:",method)
    with open(args.prompt, 'r') as config_file:
        user_config = json.load(config_file)

    config = {**default_config, **user_config}

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    with open(args.input_file, 'r') as file:
        data = json.load(file)
        sampled_data = random.sample(data, 200)

    mean_score = calculate_metric_and_output_file_llama(generator, sampled_data, config, args.output, model_name, method)

    print(f"Mean score: {mean_score}")

if __name__ == "__main__":
    main()
