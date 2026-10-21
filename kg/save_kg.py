import numpy as np
import argparse
import asyncio
import pickle
import json
import time
import os
import re

### Define Arguments ###
parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live", help="Target dataset.")
parser.add_argument("-s", type=str, default="lite", help="Target split.")
parser.add_argument("-n", type=int, default=10, help="Number of seed issues.")
parser.add_argument("-hop", type=int, default=1, help="Number of hops.")
parser.add_argument("-m", type=str, default="gpt-4o-mini", help="LLM type.")
parser.add_argument("-w", type=int, default=32, help="Number of workers.")
args = parser.parse_args()
### Arguments ###

def remove_duplicate(x):
    return list(dict.fromkeys(x))

### Load Dataset ###
name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)
### Load Dataset ###

### Logging ###
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

os.makedirs(f"./logs/{name_data}", exist_ok=True)
file_handler = logging.FileHandler(f"./logs/{name_data}/save_kg.log", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
### Logging ###

for idx in data:
    instance = data[idx]
    reponame = instance["repo"]

    str_inst = f"### Saving Knowledge Graph for {instance['instance_id']} ({idx}/{len(data)}) ###"
    logger.info("#" * len(str_inst))
    logger.info(str_inst)
    logger.info("#" * len(str_inst))
    start = time.time()

    with open(f"./res/{instance['instance_id']}/triplet_schema_seed_{args.n}_hop_{args.hop}.txt", "r") as f:
        triplets_schema = [tuple(line.strip().split("\t")) for line in f.readlines()]
    with open(f"./res/{instance['instance_id']}/dict_issue.json", "r") as f:
        dict_issue = json.load(f)
    with open(f"./res/{instance['instance_id']}/dict_commit.json", "r") as f:
        dict_commit = json.load(f)

    with open(f"./res/{instance['instance_id']}/text_issue.json", "r") as f:
        text_issue = json.load(f)
    with open(f"./res/{instance['instance_id']}/text_commit.json", "r") as f:
        text_commit = json.load(f)

    with open(f"./res/{instance['instance_id']}/summary_issue.pkl", "rb") as f:
        summary_issue = pickle.load(f)
    with open(f"./res/{instance['instance_id']}/summary_commit.pkl", "rb") as f:
        summary_commit = pickle.load(f)

    with open(f"./res/{instance['instance_id']}/entity.pkl", "rb") as f:
        entity = pickle.load(f)

    with open(f"./res/{instance['instance_id']}/triplets.pkl", "rb") as f:
        triplets = pickle.load(f)

    triplets_all = []
    for key in triplets:
        if triplets[key].get("triplets"):
            triplets_all += triplets[key]["triplets"]
    triplets_all += triplets_schema
    triplets_all = remove_duplicate(triplets_all)

    triplets = []
    for h, r, t in triplets_all:
        if h.lower().startswith("issue"):
            h = f"ISSUE:{h[6:]}"
        elif h.lower().startswith("pr"):
            h = f"PR:{h[3:]}"
        elif h.lower().startswith("commit"):
            h = f"COMMIT:{h[7:13]}"

        if t.lower().startswith("issue"):
            t = f"ISSUE:{t[6:]}"
        elif t.lower().startswith("pr"):
            t = f"PR:{t[3:]}"
        elif t.lower().startswith("commit"):
            t = f"COMMIT:{t[7:13]}"

        triplets.append((h, r.upper(), t))
    triplets = sorted(remove_duplicate(triplets))
    triplets = [(h.replace('\t', ' '), r.replace('\t', ' '), t.replace('\t', ' ')) for h, r, t in triplets]

    nodes = sorted(set([h for h, _, _ in triplets] + [t for _, _, t in triplets]))

    name_map = dict()
    for issue in dict_issue:
        if dict_issue[issue]:
            if dict_issue[issue]["is_pr"]:
                key = f"PR:{issue}"
                name = f"PR:{issue} ({dict_issue[issue].get('title')})"
            else:
                key = f"ISSUE:{issue}"
                name = f"ISSUE:{issue} ({dict_issue[issue].get('title')})"
            name_map[key] = name

    for sha in dict_commit:
        if dict_commit[sha]:
            key = f"COMMIT:{sha[:6]}"
            message = dict_commit[sha].get("message")
            if message:
                name = f"COMMIT:{sha[:6]} ({message.strip().splitlines()[0].strip()})"
            else:
                name = f"COMMIT:{sha[:6]}"
            name_map[key] = name
        
    for n in nodes:
        if n not in name_map:
            name_map[n] = n

    text_sha = dict()
    for sha in text_commit:
        text_sha[sha[:6]] = text_commit[sha]
    summary_sha = dict()
    for sha in summary_commit:
        summary_sha[sha[:6]] = summary_commit[sha]

    metadata = dict()
    for node in nodes:
        if node.startswith("ISSUE:"):
            issue = node[6:]
            text = text_issue.get(issue)
            summary = summary_issue.get(issue).get("summary") if summary_issue.get(issue) else ""
            metadata[node] = {"text": text, "summary": summary}
        elif node.startswith("PR:"):
            issue = node[3:]
            text = text_issue.get(issue)
            summary = summary_issue.get(issue).get("summary") if summary_issue.get(issue) else ""
            metadata[node] = {"text": text, "summary": summary}
        elif node.startswith("COMMIT:"):
            sha = node[7:]
            text = text_sha.get(sha)
            summary = summary_sha.get(sha).get("summary") if summary_sha.get(sha) else ""
            metadata[node] = {"text": text, "summary": summary}

    DIR_SAVE = f"./kg/{instance['instance_id']}/"
    os.makedirs(DIR_SAVE, exist_ok=True)

    with open(DIR_SAVE + "nodes.txt", "w") as f:
        for node in nodes:
            f.write(node + "\n")

    with open(DIR_SAVE + "edges.txt", "w") as f:
        for src_id, edge_type, tgt_id in triplets:
            f.write(f"{src_id}\t{edge_type}\t{tgt_id}\n")

    with open(DIR_SAVE + "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with open(DIR_SAVE + "name_map.pkl", "wb") as f:
        pickle.dump(name_map, f)