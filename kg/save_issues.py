import subprocess
import argparse
import pickle
import json
import time
import os

### Arguments ###
parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live", help="Target dataset.")
parser.add_argument("-s", type=str, default="lite", help="Target split.")
args = parser.parse_args()
### Arguments ###

### Load Dataset ###
name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)
### Load Dataset ###

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

os.makedirs(f"./logs/{name_data}", exist_ok=True)
file_handler = logging.FileHandler(f"./logs/{name_data}/save_issues.log", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def get_all_issues_fast(repo_full_name):
    cmd = [
        "gh", "api", 
        f"repos/{repo_full_name}/issues?state=all&sort=created&direction=asc&per_page=100",
        "--paginate",
        "--jq", ".[]"
    ]
    
    logger.info(f"Fetching issues for {repo_full_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.info(f"Error: {result.stderr}")
        quit()

    issue_dict = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            item = json.loads(line)
            issue_dict[int(item['number'])] = item
            
    return issue_dict

repos = []
for idx in data:
    instance = data[idx]
    if instance["repo"] not in repos:
        repos.append(instance["repo"])

for idx, reponame in enumerate(repos):
    file_path = f"./issues/{reponame.replace('/', '__')}.jsonl"
    if os.path.exists(file_path):
        logger.info(f"Skipping {reponame}, already exists.")
        continue

    str_inst = f"### Extracting Issues for {reponame} ({idx} / {len(repos)}) ###"
    logger.info("#" * len(str_inst))
    logger.info(str_inst)
    logger.info("#" * len(str_inst))

    start = time.time()
    issue_dict = get_all_issues_fast(reponame)

    logger.info(f"Extracted all issues in {time.time() - start:.1f}s.")
    os.makedirs(f"./issues", exist_ok=True)
    with open(file_path, "w") as f:
        for i in sorted(issue_dict.keys()):
            f.write(json.dumps(issue_dict[i], ensure_ascii=False) + "\n")