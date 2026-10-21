from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

import os
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
parser.add_argument("-s", type=str, default="lite")
args = parser.parse_args()

### Reproducibility/PyTorch Settings ###
import torch
import random

OMP_NUM_THREADS = 8
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)
# torch.autograd.set_detect_anomaly(True) # For debug only
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()
### Reproducibility/PyTorch Settings ###

name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)

import requests

def embed_query(instruct, query):
    input_query = f"Instruct: {instruct.strip()}\nQuery: {query.strip()}"
    response = requests.post("http://localhost:9000/embed", json={"sentences": [input_query]})
    return np.array(response.json()["embeddings"][0])

for i in tqdm(range(len(data))):
    instance_info = data[i]

    args.i = instance_info["instance_id"]
    problem_statement = instance_info.get("problem_statement")

    DIR_SAVE = f"./../cg/cg/{args.i}/"
    if not os.path.exists(DIR_SAVE + f"emb_codes.npy"):
        continue
    code_embeddings = np.load(DIR_SAVE + f"emb_codes.npy")

    with open(DIR_SAVE + "nodes.txt", "r") as f:
        nodes = [line.strip() for line in f.readlines()]
    node2id = {node: idx for idx, node in enumerate(nodes)}

    query = problem_statement.strip()
    query_embedding = embed_query("Given a github issue, identify the code that needs to be changed to fix the issue.", query.strip())
    sim = np.sum(code_embeddings * query_embedding, -1)
    ranks = np.argsort(-sim)

    res = []
    for idx in ranks:
        if not nodes[idx].startswith("FUNC:"):
            continue
        res.append((sim[idx], nodes[idx]))

    os.makedirs(f"./res_emb/{name_data}/", exist_ok=True)
    with open(f"./res_emb/{name_data}/{args.i}.pkl", "wb") as f:
        pickle.dump(res, f)