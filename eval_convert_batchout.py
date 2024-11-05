import sys, os
import json

def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cur_jd = json.loads(line.strip())
            data[cur_jd["custom_id"].zfill(6)] = cur_jd["response"]["body"]["choices"][0]["message"]["content"]
    return data


file_path = sys.argv[1]
model = file_path.split("_eval_")[-1].split("_output")[0]
output_data = load_jsonl(file_path)

ori_file_path = file_path.split("_eval_")[0] + ".json"

with open(ori_file_path, "r") as f:
    jd = json.load(f)

if "suffix" in jd:
    jd.pop("suffix")
    
jd_save = {}
for key, content in jd.items():
    goal = content["goal"]
    target = content["target"]
    translations = content["translations"]
    response_ls = []
    for j in range(len(translations)):
        output_id = f"{int(key):03}{j:03}"
        if output_id in output_data.keys():
            response = output_data[output_id]
            response_ls.append(response)
    jd_save[key] = {"goal": goal, "target": target, "response":response_ls}

save_path = ori_file_path.replace(".json", f"_eval_{model}.json")
with open(save_path, "w") as f:
    json.dump(jd_save, f, indent=4)
print(save_path)