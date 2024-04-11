import argparse
import json
import torch
import os
import random
from utils import calculate_metric_and_output_file_ptuning
from transformers import AutoConfig, AutoModel, AutoTokenizer

def load_model_and_checkpoint(model_dir, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.pre_seq_len = 128
    config.prefix_projection = False

    model = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))

    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    model = model.half().cuda()
    model.transformer.prefix_encoder.float().cuda()
    model = model.eval()

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Ptuning Runner")
    parser.add_argument("--ckpt_dir", type=str, default="models/chatglm-6b", help="Path to model directory")
    parser.add_argument("--ptuning_dir", type=str, default="models/ptuning/checkpoint-1000", help="Path to model checkpoint directory")
    parser.add_argument("--input_file", type=str, default="data/zh/robustness/HITD_noise1.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="outputs", help="Path to output directory")

    args = parser.parse_args()

    data_name = os.path.basename(args.input_file)
    print("data name",data_name)

    # 加载模型和checkpoint
    model, tokenizer = load_model_and_checkpoint(args.ckpt_dir, args.ptuning_dir)

    with open(args.input_file, 'r') as file:
        data = json.load(file)
        sampled_data = random.sample(data, 200)


    mean_score = calculate_metric_and_output_file_ptuning(model, tokenizer, sampled_data, args.output, data_name)

    print(f"Mean score: {mean_score}")


if __name__ == "__main__":
    main()
