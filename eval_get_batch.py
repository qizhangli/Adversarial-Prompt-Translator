import sys
import json
source_path = sys.argv[1]
model = sys.argv[2]

with open(source_path, "r") as f:
    jd = json.load(f)

if "suffix" in jd:
    suffix = jd.pop("suffix")

half = 0
save_dir = source_path.replace(".json", "_eval_{}.jsonl".format(model))
with open(save_dir, "w") as f:
    for key, value in jd.items():
        prompts = value["translations"]
        c = 0
        for prompt in prompts:
            curr_dict = {"custom_id": f"{int(key):03}{c:03}",
                        "method": "POST", 
                        "url": "/v1/chat/completions", 
                        "body": {"model": f"{model}", 
                                 "temperature": 0.1,
                                 "max_tokens": 150,
                                 "messages": [{"role": "user", "content": prompt}]}}
            f.write(json.dumps(curr_dict)+"\n")
            c+=1
print(save_dir)
