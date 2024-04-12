# Middle Code Prediction(MCP)

The code for the paper "Middle Code Prediction: a lightweight scheme for Code Generation in the Robotics Domain", which proposes a lightweight method of Middle Code Prediction (MCP) aimed at addressing the imbalance between prediction accuracy and efficiency in rare data domains with large language models. We validated MCP on a Hospital Item Transport Dataset[HITD](data/README.md) and found that it achieved a new balance in accuracy and efficiency. Furthermore, it can generalize to different tasks of robot code generation.
![Introduction of HITD](https://github.com/Ghbbbbb/MCP/blob/main/assets/MCP.png)
## Clone
Clone this repo and install requirements. 

    git clone https://github.com/Ghbbbbb/MCP.git
    # python3.10
    pip install -r requirements.txt

## Models
Download Llama2 series models（provided by [codellama Project](https://github.com/meta-llama/codellama)）and ChatGLM2 models（provided by [ChatGLM2-6B Project](https://github.com/THUDM/ChatGLM2-6B)）

Then download the P-tuning v2 weight file:
- [BaiDu Cloud Drive](https://pan.baidu.com/s/1cuTCQmiQzp33NFfk682jFA) with code: 78wk

Put all models in the `models`  directory and the structure of the file directory tree is shown below:

```
├── models
│   ├── chatglm-6b
│   ├── CodeLlama-7b-Instruct
│   ├── llama
│   ├── Llama-7b
│   └── ptuning
```

## Run

- llama
```
torchrun run_llama.py  [--ckpt_dir *** --tokenizer_path *** --input_file *** --prompt ***]
```

- gpt
```
export OPENAI_KEY = [YOUR_KEY]
torchrun run_gpt.py  [--model *** --input_file *** --prompt ***]
```
- P-tuning v2
```
torchrun run_ptuning.py  [--ckpt_dir *** --ptuning_dir *** --input_file ***]
```
-  Zero-shot
```
python run_gsm8k_zs.py
python run_aqua_zs.py
...
```

The prediction file will be dumped in the outputs/ folder, let's say gsm8K_s0_e-1_11_17_10_20.jsonl, or gsm8K_sc_s0_e-1_11_08_21_14.jsonl, or  gsm8K_zs_s0_e-1_11_19_09_55.jsonl.

- Evaluation
```
cd outputs
python compare.py --inputs CodeLlama-7b-Instruct_FCP_5shot_HITD.json
python compare.py --inputs CodeLlama-7b-Instruct_MCP_5shot_HITD.json
python compare.py --inputs gpt-3.5-turbo-16k_FCP_5shot_HITD.json
....
```

## 5-shot Main Results(MCP vs FCP)
1. Llama-7b
- Output: outputs/Llama-7b_MCP_5shot_HITD.json
- Mean Score: 0.278

- Output: outputs/Llama-7b_FCP_5shot_HITD.json
- Mean Score: 0.206

2. CodeLlama-7b-Instruct
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD.json
- Mean Score: 0.733

- Output: outputs/CodeLlama-7b-Instruct_FCP_5shot_HITD.json
- Mean Score: 0.402

3. Gpt3.5-turbo-175b
- Output: outputs/gpt-3.5-turbo-16k_MCP_5shot_HITD.json
- Mean Score: 0.847

- Output: outputs/gpt-3.5-turbo-16k_FCP_5shot_HITD.json
- Mean Score: 0.640



## 5-shot Robustness Result(MCP Vs Ptuningv2)
1. No noise
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_no_noise.json
- Mean Score: 0.695

- Output: outputs/chatglm2-6b_ptuning_HITD_no_noise.json
- Mean Score: 0.938

2. Noise1
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_noise1.json
- Mean Score: 0.639(↓0.056)

- Output: outputs/chatglm2-6b_ptuning_HITD_noise1.json
- Mean Score: 0.913(↓0.025)

3. Noise2
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_noise2.json
- Mean Score: 0.741(↑0.046)

- Output: outputs/chatglm2-6b_ptuning_HITD_noise2.json
- Mean Score: 0.629(↓0.309)

4. Noise3
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_noise3.json
- Mean Score: 0.703(↑0.008)

- Output: outputs/chatglm2-6b_ptuning_HITD_noise3.json
- Mean Score: 0.681(↓0.257)