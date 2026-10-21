import argparse
import pickle
import torch
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
parser.add_argument("-s", type=str, default="lite")
args = parser.parse_args()

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True, device="cuda").to(torch.bfloat16)
model._first_module().auto_model.config.use_cache = False

tokenizer = model.tokenizer
model.max_seq_length = 4096

name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)

for idx in data:
    instance = data[idx]

    DIR_SAVE = f"./kg/{instance['instance_id']}/"
    if os.path.exists(DIR_SAVE + f"emb_codes.npy"):
        print(f"{instance['instance_id']} already processed. skip.")
        continue

    str_inst = f"### Encoding {instance['instance_id']} ({idx}/{len(data)}) ###"
    print("#" * len(str_inst))
    print(str_inst)
    print("#" * len(str_inst))

    with open(DIR_SAVE + "nodes.txt", "r") as f:
        nodes = [line.strip() for line in f.readlines()]
    with open(DIR_SAVE + "edges.txt", "r") as f:
        edges = [line.strip().split("\t") for line in f.readlines()]
    with open(DIR_SAVE + "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    with open(DIR_SAVE + "name_map.pkl", "rb") as f:
        name_map = pickle.load(f)

    codes = []
    for node in tqdm(nodes):
        if node in metadata:
            context = metadata[node].get("text", node)
            if context:
                codes.append(context.strip())
            elif node in name_map:
                codes.append(name_map[node].strip())
            else:
                codes.append(node.strip())
        elif node in name_map:
            codes.append(name_map[node].strip())
        else:
            codes.append(node.strip())

    dict_codes = dict()
    for node, content in zip(nodes, codes):
        dict_codes[node] = content
    
    import json
    with open(DIR_SAVE + "dict_codes.json", "w") as f:
        json.dump(dict_codes, f, indent=4)

    edges = [(name_map.get(h, h), r.upper(), name_map.get(t, t)) for h, r, t in edges]
    nodes = [name_map.get(n, n).strip() for n in nodes]

    emb_nodes = model.encode(nodes, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    emb_edges = model.encode([f"{h.strip()} {r.strip()} {t.strip()}" for h, r, t in edges], batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    emb_codes = model.encode(codes, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    np.save(DIR_SAVE + f"emb_nodes.npy", emb_nodes)
    np.save(DIR_SAVE + f"emb_edges.npy", emb_edges)
    np.save(DIR_SAVE + f"emb_codes.npy", emb_codes)