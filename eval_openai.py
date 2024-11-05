from openai import OpenAI
import argparse


base_url = "your_own"
API_KEY = "your_own"
client = OpenAI(base_url=base_url, api_key=API_KEY)


def submit(file_dir):
    res = client.files.create(
                                file=open(file_dir, "rb"),
                                purpose="batch"
                                )
    print()
    print(res)
    res = client.batches.create(
                                    input_file_id=res.id,
                                    endpoint="/v1/chat/completions",
                                    completion_window="24h",
                                    metadata={
                                        "description": f"{file_dir}",
                                        # "_use_self_hosted": True
                                    }
                                )
    print()
    print(res)


def view():
    res = client.batches.list(
        after="",
        limit=20
    )
    return res

def process_view(res):
    import pandas as pd
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd_dict = {"id":[], "status": [], "description": []}
    for i in range(len(res.data)):
        pd_dict["id"].append(i)
        pd_dict["status"].append(res.data[i].status)
        pd_dict["description"].append(res.data[i].metadata["description"])
    return pd.DataFrame(pd_dict)

def download(file_id):
    content = client.files.content(file_id)
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--view", default=False, action="store_true")
    parser.add_argument("--submit", default=False, action="store_true")
    parser.add_argument("--download", default=None, type=str)
    parser.add_argument("--cancel", default=None, type=str)

    args = parser.parse_args()
    
    if args.view:
        assert args.submit == False
        res = view()
        print()
        print(process_view(res).to_string(index=False))
    
    if args.submit:
        assert args.log_path != None
        submit(args.log_path)
        print("submitted.")
    
    if args.download != None:
        res = view()
        ids = [int(t) for t in args.download.split(",")]
        for id_down in ids:
            info = res.data[id_down]
            content = download(info.output_file_id)
            save_dir = info.metadata["description"].replace(".jsonl", "_output.jsonl")
            with open(save_dir, 'wb') as f:
                f.write(content.content)
            print(save_dir)
    
    if args.cancel != None:
        res = view()
        ids = [int(t) for t in args.cancel.split(",")]
        for id_ in ids:
            client.batches.cancel(res.data[id_].id)