# Adversarial-Prompt-Translator

This repository contains a PyTorch implementation for our paper [***Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation***](https://arxiv.org/abs/2410.11317). 

## Environments
* Python 3.8.8
* PyTorch 2.4.0
* transformers 4.44.2
* vllm 0.5.4
* openai 1.44.1

## Usage
To generate adversarial prompts using Llama-3.1-8B as the translator LLM on the HarmBench dataset, run:
```
translate.py --translator llama3.1-8b --dataset harmbench
```
The adversarial prompts will be saved in ```results/trans_harmbench_llama3.1-8b.json```.

For evaluation, we use OpenAI's Batch API.
To evaluate the performance of attacking a GPT/Claude model, for instance, gpt-4o-mini, submit the adversarial prompts as a batch by running:
```
logpath=results/trans_harmbench_llama3.1-8b.json model=gpt-4o-mini bash eval_submit.sh
```
You can track the status of the generation by running:
```
python3 eval_openai.py --v
```
The output will be like:
``` 
id   status        description
0    in_progress   results/trans_harmbench_llama3.1-8b_eval_gpt-4o-mini.jsonl
...  ...           ...
```
The "id" is used for identifying different submissions. Once the status turns to "completed", download the result file by running:
```
python3 eval_openai.py --d {id}
```
Then, to evaluate the attack success rate using the classifier of HarmBench, run:
```
bash eval_harmbench.py
```
The results will be saved in ```results/trans_harmbench_llama3.1-8b_eval_gpt-4o-mini_eval.json```


## Citation
Please cite our work in your publications if it helps your research:

```
@article{li2024deciphering,
         title={Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation},
         author={Li, Qizhang and Yang, Xiaochen and Zuo, Wangmeng and Guo, Yiwen},
         journal={arXiv preprint arXiv:2410.11317},
         year={2024}
}
```
