import json
from collections import Counter
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)
args = parser.parse_args()


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def bleu_score(reference, candidate, n=4):
    # 计算n-gram精确匹配的准确度
    precision = 1.0
    for i in range(1, n + 1):
        reference_ngram_counts = Counter(zip(*[reference[j:] for j in range(i)]))
        candidate_ngram_counts = Counter(zip(*[candidate[j:] for j in range(i)]))
        common_ngrams = sum((candidate_ngram_counts & reference_ngram_counts).values())
        total_ngrams = sum(candidate_ngram_counts.values())
        
        # 避免出现除以零的情况
        if total_ngrams == 0:
            precision *= 0.0
        else:
            precision *= common_ngrams / total_ngrams

    # 计算翻译长度惩罚
    reference_length = len(reference)
    candidate_length = len(candidate)
    length_penalty = min(1, math.exp(1 - reference_length / candidate_length))

    # 计算最终BLEU分数
    bleu = length_penalty * precision ** (1/n)

    return bleu

def calculate_score(json_file):
    data = load_json_file(json_file)
    total_samples = len(data)
    correct_count = 0
    bleu_sum = 0

    for sample in data:
        if 'final_code' in sample and 'answear' in sample:
            final_code = sample['final_code'].strip()
            answear = sample['answear'].strip()

            if final_code == answear:
                correct_count += 1

            if final_code:
                bleu_score_value = bleu_score(answear, final_code)
            else:
                bleu_score_value = 0.5
            bleu_sum += bleu_score_value

    accuracy = correct_count / total_samples
    avg_bleu = bleu_sum / total_samples
    mean_socre = 0.5*(accuracy + avg_bleu)

    return accuracy, avg_bleu, mean_socre


accuracy, avg_bleu, mean_socre = calculate_score(args.inputs)

# 输出准确率和平均 BLEU 分数
print(f"Accuracy: {accuracy:.3f}")
print(f"BLEU Score: {avg_bleu:.3f}")
print(f"Mean Score: {mean_socre:.3f}")
