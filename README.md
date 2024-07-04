# Middle Code Prediction
This is code repository for the paper **" Middle Code Prediction: Enhancing Code Generation for Uncommon Programming Languages in Robotics"**.  

We introduce Middle Code Prediction (MCP), a scheme that allows LLMs to adapt to various low-level code prediction tasks through the injection of prompts at different stages. We validated MCP on a Hospital Item Transport Dataset[(HITD)](data/README.md) and found that it can improve the task mean accuracy of various baseline models to varying degrees, with an overall improvement of 32%.

In the following experiments, we evaluated the ability of MCP, FCP, and P-tuningv2 to generate low-level code using LLMs. To ensure a fair comparison with fine-tuning methods, we assessed the single-round code generation performance of each method.


![Three-stage Framework of MCP](https://github.com/Ghbbbbb/MCP/blob/main/assets/MCP.png)



## Clone
Clone this repo and install requirements. 

    git clone https://github.com/Ghbbbbb/MCP.git
    # python3.10
    pip install -r requirements.txt

## Models
Download Llama2 series models（provided by [codellama Project](https://github.com/meta-llama/codellama)）and ChatGLM2 models（provided by [ChatGLM2-6B Project](https://github.com/THUDM/ChatGLM2-6B)）

Then download the P-tuning v2 weight file(fine-tuned on final code):
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

The prediction file will be dumped in the outputs/ folder, let's say CodeLlama-7b-Instruct_FCP_5shot_HITD.json, CodeLlama-7b-Instruct_MCP_5shot_HITD.json

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
3. Gpt3.5-turbo-16k
- Output: outputs/gpt-3.5-turbo-16k_MCP_5shot_HITD.json
- Mean Score: 0.847  
- Output: outputs/gpt-3.5-turbo-16k_FCP_5shot_HITD.json
- Mean Score: 0.640  

## 5-shot Robustness Result(MCP vs P-tuningv2)
1. No noise
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_no_noise.json
- Mean Score: 0.695  
- Output: outputs/chatglm2-6b_ptuning_HITD_no_noise.json
- Mean Score: 0.938  
2. Noise1
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_noise1.json
- Mean Score: 0.639(**↓8.06%**)  
- Output: outputs/chatglm2-6b_ptuning_HITD_noise1.json
- Mean Score: 0.913(**↓2.67%**)  
3. Noise2
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_noise2.json
- Mean Score: 0.741(**↑6.62%**)  
- Output: outputs/chatglm2-6b_ptuning_HITD_noise2.json
- Mean Score: 0.629(**↓32.94%**)  
4. Noise3
- Output: outputs/CodeLlama-7b-Instruct_MCP_5shot_HITD_noise3.json
- Mean Score: 0.703(**↑1.15%**)  
- Output: outputs/chatglm2-6b_ptuning_HITD_noise3.json
- Mean Score: 0.681(**↓27.40%**)