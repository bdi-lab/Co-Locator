import argparse
import pickle

from repo_processor import RepoProcessor
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
    parser.add_argument("-s", type=str, default="lite")
    args = parser.parse_args()

    name_data = args.d.strip().split("/")[1]
    with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
        data = pickle.load(f)

    for idx in data:
        instance = data[idx]

        repo = instance["repo"]
        base_commit = instance["base_commit"]

        metadata_path = Path(f"./cg/{instance['instance_id']}/metadata.pkl")
        if metadata_path.is_file():
            print(f"{instance['instance_id']} already processed. skip.")
            continue

        str_inst = f"### Extracting Code Graph for {instance['instance_id']} ({idx}/{len(data)}) ###"
        print("#" * len(str_inst))
        print(str_inst)
        print("#" * len(str_inst))
        repoProcessor = RepoProcessor(instance, repo, base_commit)
        repoProcessor.extract_cg()