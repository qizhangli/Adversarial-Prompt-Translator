import torch
from utils import get_data, process_output, MODEL_INFO, SUFFIXES
import json
import argparse
from get_query import *
from vllm import LLM, SamplingParams


class AskLLM(object):
    def __init__(self, model_path):
        self.model = LLM(
            model=model_path,
            enable_prefix_caching=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=10000
        )
    def ask_llm(self, message, max_tokens=1000):
        params = {
            "temperature":1,
            "top_p":0.9,
            "max_tokens": max_tokens,
        }
        sampling_params = SamplingParams(**params)
        outputs = self.model.generate(message, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]


def main(args, model, data_pairs, suffix):
    save_dict = {"suffix": suffix}

    for i in range(len(data_pairs)):
        goal, _, target = data_pairs[i]
        
        #####################
        # rephrase
        #####################
        query = get_query_rephrase(goal, target, sep=args.sep)
        rephrased_prompts = ["1. \"" + t for t in model.ask_llm([query]*2)]
        rephrased_prompts = process_output(rephrased_prompts)[:10]
        
        #####################
        # interpretation
        #####################
        queries = [get_query_interpretation(t, suffix, target, sep=args.sep) for t in rephrased_prompts]
        interpretations = ["1. " + t for t in model.ask_llm(queries)]
        
        #####################
        # translation
        #####################
        queries = [get_query_translate(rephrased_prompts[t], suffix, target, interpretations[t], sep=args.sep) for t in range(len(rephrased_prompts))]
        translations = [t.split("\"")[0] for t in model.ask_llm(queries)]
        
        save_dict[i] = {"goal": goal,
                        "rephrased_prompt": rephrased_prompts,
                        "target": target,
                        "suffix": suffix,
                        "interpretations": interpretations,
                        "translations": translations}
        
        with open(args.save_dir, "w") as f:
            json.dump(save_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--translator", type=str, default="llama3.1-8b")
    parser.add_argument("--dataset", type=str, default="harmbench", choices=["harmbench", "advbench"])
    args = parser.parse_args()
    
    args.save_dir = "results/trans_{}_{}.json".format(args.dataset, args.translator)

    suffix = SUFFIXES[0].strip() # 0 for concatenation, 1 for universal suffix generated using Llama-3.1-8b on HarmBench, 2 for universal suffix collected from HarmBench's playground.
    data_pairs = get_data(args.dataset)
    model_path = MODEL_INFO[args.translator]["model_path"]
    args.sep = MODEL_INFO[args.translator]["sep"]
    
    model = AskLLM(model_path=model_path)
    main(args, model, data_pairs, suffix)

        
        

