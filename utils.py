from collections import Counter
import math
import json
import os
from tqdm import tqdm


def process_input_llama(generator, user_input, config) -> str:

    instructions = [
        [
            {"role": "system", "content": config["system_msg"]},
            {"role": "user", "content": f"{config['user_input_prefix']}{user_input}{config['user_input_suffix']}"}
        ]
    ]
    result = generator.chat_completion(
        instructions,
        max_gen_len=config["max_gen_len"],
        temperature=config["temperature"],
        top_p=config["top_p"],
    )

    generated_text = result[0]['generation']['content']

    if config["type"] =="MCP":
        generated_code = extract_code_MCP(generated_text)
        expand = expand_code(generated_code)
    else:
        expand = extract_code_FCP(generated_text)
    return generated_text,expand


def process_input_gpt(client, model_name, user_input, config) -> str:

    instructions = [
            {"role": "system", "content": config["system_msg"]},
            {"role": "user", "content": f"{config['user_input_prefix']}{user_input}{config['user_input_suffix']}"}
    ]
    result = client.chat.completions.create(
        model = model_name,
        messages=instructions,
        max_tokens = config["max_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"]
    )

    generated_text = result.choices[0].message.content.strip()
    if config["type"] =="MCP":
        generated_code = extract_code_MCP(generated_text)
        expand = expand_code(generated_code)
    else:
        expand = extract_code_FCP(generated_text)

    return generated_text,expand


def calculate_metric_and_output_file_llama(generator, data, config, output_dir, model_name, method) -> tuple:
    output_filename = f"{model_name}_{method}"
    output_path = os.path.join(output_dir, output_filename)

    output_data = []

    correct_predictions = 0
    bleu_sum = 0
    total_items = len(data)

    with tqdm(total=total_items) as pbar:
        for item in data:
            content = item.get("content", "")
            answer = item.get("summary", "")

            if content:
                user_input = f'"{content}"'
                generated_text,predicted_code = process_input_llama(generator, user_input, config)

                if config["type"] == "MCP":
                    output_item = {
                        "content": content,
                        "middle_code": generated_text, 
                        "final_code": predicted_code, 
                        "answear": answer
                    }
                else:
                    output_item = {
                        "content": content,
                        "final_code": predicted_code, 
                        "answear": answer
                    }

                output_data.append(output_item)

                # 计算准确率和 BLEU 分数
                if predicted_code == answer:
                    correct_predictions += 1

                if predicted_code:
                    bleu_score_value = bleu_score(answer, predicted_code)
                else:
                    bleu_score_value = 0.5
                bleu_sum += bleu_score_value

                # 更新进度条
                pbar.update(1)

        # 将输出数据写入到 JSON 文件中
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, indent=1, ensure_ascii=False)

        print(f"Output written to: {output_path}")

        accuracy = correct_predictions / total_items
        avg_bleu_score = bleu_sum / total_items
        mean_score = 0.5*(accuracy + avg_bleu_score)
        print("accuracy:",accuracy)
        print("bleu:",avg_bleu_score)

        return mean_score


def calculate_metric_and_output_file_gpt(client, data, config, output_dir, model_name, method) -> tuple:
    output_filename = f"{model_name}_{method}"
    output_path = os.path.join(output_dir, output_filename)

    output_data = []

    correct_predictions = 0
    bleu_sum = 0
    total_items = len(data)

    with tqdm(total=total_items) as pbar:
        for item in data:
            content = item.get("content", "")
            answer = item.get("summary", "")

            if content:
                user_input = f'"{content}"'
                generated_text,predicted_code = process_input_gpt(client, model_name, user_input, config)

                output_item = {
                    "content": content,
                    "middle_code": generated_text, 
                    "final_code": predicted_code, 
                    "answear": answer
                }

                output_data.append(output_item)

                if predicted_code == answer:
                    correct_predictions += 1

                if predicted_code:
                    bleu_score_value = bleu_score(answer, predicted_code)
                else:
                    bleu_score_value = 0.5
                bleu_sum += bleu_score_value

                pbar.update(1)


        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, indent=1, ensure_ascii=False)

        print(f"Output written to: {output_path}")

        accuracy = correct_predictions / total_items
        avg_bleu_score = bleu_sum / total_items
        mean_score = 0.5*(accuracy + avg_bleu_score)
        print("accuracy:",accuracy)
        print("bleu:",avg_bleu_score)

        return mean_score


def calculate_metric_and_output_file_ptuning(model, tokenizer, data, output_dir, data_name) -> tuple:
    output_filename = f"chatglm2-6b_ptuning_{data_name}"
    output_path = os.path.join(output_dir, output_filename)

    output_data = []

    correct_predictions = 0
    bleu_sum = 0
    total_items = len(data)

    with tqdm(total=total_items) as pbar:
        for item in data:
            content = item.get("content", "")
            answer = item.get("summary", "")

            if content:
                user_input = f'"{content}"'
                predict, history = model.chat(tokenizer, content, history=[])

                output_item = {
                    "content": content,
                    "final_code": predict, 
                    "answear":answer
                }

                output_data.append(output_item)

                if predict == answer:
                    correct_predictions += 1

                if predict:
                    bleu_score_value = bleu_score(answer, predict)
                    bleu_sum += bleu_score_value

                pbar.update(1)


        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, indent=1, ensure_ascii=False)

        print(f"\nOutput written to: {output_path}")

        accuracy = correct_predictions / total_items
        avg_bleu_score = bleu_sum / total_items
        
        mean_score = 0.5*(accuracy + avg_bleu_score)
        print("accuracy:",accuracy)
        print("bleu:",avg_bleu_score)

        return mean_score


def bleu_score(reference, candidate, n=4):

    precision = 1.0
    for i in range(1, n + 1):
        reference_ngram_counts = Counter(zip(*[reference[j:] for j in range(i)]))
        candidate_ngram_counts = Counter(zip(*[candidate[j:] for j in range(i)]))
        common_ngrams = sum((candidate_ngram_counts & reference_ngram_counts).values())
        total_ngrams = sum(candidate_ngram_counts.values())
        
        if total_ngrams == 0:
            precision *= 0.0
        else:
            precision *= common_ngrams / total_ngrams

    reference_length = len(reference)
    candidate_length = len(candidate)
    length_penalty = min(1, math.exp(1 - reference_length / candidate_length))

    bleu = length_penalty * precision ** (1/n)

    return bleu


def expand_code(short_code):
    try:
        start_blocks = short_code.split("END")

        full_code = ""

        for start_block in start_blocks:
            if not start_block:
                continue
            parts = start_block.split()

            X = int(parts[0][6:-1])
            Y_list = parts[1:]

            full_code += f'WHILE #GP({Y_list[0]}) < {X}'

            for Y in Y_list[:]:
                full_code += f' MOVP P = {Y} OP = 28000'

            full_code += f' ADD #GP({Y_list[0]}) 1 ENDWHILE '

        return full_code[:-1].strip()
    except Exception:
        return short_code


def extract_code_MCP(text):
    start_index = text.find("{")
    if start_index != -1:
        start_index += len("{")
        end_index = text.find("}", start_index)
        if end_index != -1:
            return text[start_index:end_index]
    return ""


def extract_code_FCP(generated_text: str) -> str:
    parts = generated_text.split("summary:")
    if len(parts) == 1:
        parts = generated_text.split("Summary:")
    if len(parts) > 1:
        code_lines = parts[-1].strip().split('\n')
        code = " ".join(code_lines)
        return code
    else:
        return ""