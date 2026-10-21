import json
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
parser.add_argument("-s", type=str, default="lite")
parser.add_argument("-i", type=int, default=0)
parser.add_argument("-m", type=str, default="qwen2.5")
parser.add_argument("-limit_kg", type=int, default=10)
parser.add_argument("-limit_loc", type=int, default=60)
parser.add_argument("-n", type=int, default=600)
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()

### Reproducibility/PyTorch Settings ###
import torch
import random

OMP_NUM_THREADS = 8
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
# torch.autograd.set_detect_anomaly(True) # For debug only
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()
### Reproducibility/PyTorch Settings ###

name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)

res_file = {5: 0, 10: 0}
res_module = {10: 0, 15: 0}
res_func = {10: 0, 15: 0}

import requests

def embed_query(instruct, query):
    input_query = f"Instruct: {instruct.strip()}\nQuery: {query.strip()}"
    response = requests.post("http://localhost:9000/embed", json={"sentences": [input_query]})
    return np.array(response.json()["embeddings"][0])

cnt = 0
for i in tqdm(range(min(len(data), args.n))):
    instance_info = data[i]
    instance = instance_info["instance_id"]
    problem_statement = instance_info["problem_statement"]

    DIR_RES = f"{name_data}/{args.m}_loc_{args.limit_loc}_kg_{args.limit_kg}_seed_{args.seed}"
    if not os.path.exists(f"res/{DIR_RES}/results_{instance}.json"):
        continue

    DIR_CG = f"./../cg/cg/{instance}/"
    code_embeddings = np.load(DIR_CG + f"emb_codes.npy")

    with open(DIR_CG + "nodes.txt", "r") as f:
        nodes = [line.strip() for line in f.readlines()]
    
    with open(f"res/{DIR_RES}/results_{instance}.json", "r") as f:
        res_raw = json.load(f)

    preds_func = res_raw["refined_predictions"]
    preds_str = "\n".join(f"- {fn}" for fn in preds_func)
    query = f"[Problem Statement]\n{problem_statement.strip()}\n\n[Predicted Functions]\n{preds_str.strip()}"

    query_embedding = embed_query("Given a github issue and functions to be modified, identify additional functions that need to be changed to fix the issue.", query.strip())
    sim = np.sum(code_embeddings * query_embedding, -1)
    ranks = np.argsort(-sim)

    preds_func = []
    for idx in ranks:
        if not nodes[idx].startswith("FUNC:"):
            continue
        preds_func.append(nodes[idx])
    preds_func = [":".join(fn.split(":")[1:3])[1:] for fn in preds_func]
    preds_func = [fn for fn in preds_func if "test" not in fn.lower()]
    preds_func = [fn for fn in preds_func if ".py" in fn.lower()]
    preds_func = [fn for fn in preds_func if len(fn.split(":")) == 2]

    preds_file, preds_module = [], []
    for pred in preds_func:
        if "test" in pred.lower():
            continue
        file, full_name = pred.split(':')
        if file not in preds_file:
            preds_file.append(file)
        module = f"{file}:{full_name.split('.')[0]}"
        if module not in preds_module:
            preds_module.append(module)

        if len(preds_file) >= 10 and len(preds_module) >= 20:
            break

    gt_data = instance_info["edit_functions"]
    gt_file = set()
    gt_module = set()
    gt_func = set(gt_data)

    for gt in gt_data:
        file, full_name = gt.split(':')
        gt_file.add(file)
        gt_module.add(f"{file}:{full_name.split('.')[0]}")

    label_file = []
    for pred in preds_file:
        if pred in gt_file:
            label_file.append(1)
        else:
            label_file.append(0)

    label_module = []
    for pred in preds_module:
        if pred in gt_module:
            label_module.append(1)
        else:
            label_module.append(0)

    label_func = []
    for pred in preds_func:
        if pred in gt_func:
            label_func.append(1)
        else:
            label_func.append(0)

    for k in [5, 10]:
        num_gt_file = min(k, len(gt_file))
        res_file[k] += int(sum(label_file[:k]) == num_gt_file)

    for k in [10, 15]:
        num_gt_func = min(k, len(gt_func))
        res_func[k] += int(sum(label_func[:k]) == num_gt_func)

        num_gt_module = min(k, len(gt_module))
        res_module[k] += int(sum(label_module[:k]) == num_gt_module)

    cnt += 1

print(f"Results on {cnt} instances:")
print("=" * 25)
print(f"[File Level]")
print(f"Hit@5 : {res_file[5] / cnt:.3f} ({res_file[5]} / {cnt})")
print(f"Hit@10 : {res_file[10] / cnt:.3f} ({res_file[10]} / {cnt})")
print("-" * 25)
print(f"[Module Level]")
print(f"Hit@10: {res_module[10] / cnt:.3f} ({res_module[10]} / {cnt})")
print(f"Hit@15: {res_module[15] / cnt:.3f} ({res_module[15]} / {cnt})")
print("=" * 25)
print(f"[Function Level]")
print(f"Hit@10: {res_func[10] / cnt:.3f} ({res_func[10]} / {cnt})")
print(f"Hit@15: {res_func[15] / cnt:.3f} ({res_func[15]} / {cnt})")
print("=" * 25)
