import argparse
import json
import random
from openai import OpenAI
import os
from utils import calculate_metric_and_output_file_gpt

def main():

    default_config_file = "config.json"
    with open(default_config_file, 'r') as config_file:
        default_config = json.load(config_file)

    parser = argparse.ArgumentParser(description="GPT-3.5 Turbo test")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI-model name")
    parser.add_argument("--input_file", type=str, default="data/zh/HITD.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--prompt", type=str, default="prompt/zh/MCP_5shot_HITD.json", help="Path to prompt file")
    args = parser.parse_args()

    method = os.path.basename(args.prompt)
    print("method:",method)

    with open(args.prompt, 'r') as config_file:
        user_config = json.load(config_file)

    config = {**default_config, **user_config}

    client = OpenAI()

    with open(args.input_file, 'r') as file:
        data = json.load(file)
        sampled_data = random.sample(data, 100) 

    mean_score = calculate_metric_and_output_file_gpt(client, sampled_data, config, args.output, args.model, method)

    print(f"Mean score: {mean_score}")

if __name__ == "__main__":
    main()

 