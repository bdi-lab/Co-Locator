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

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True).to(torch.bfloat16)
model._first_module().auto_model.config.use_cache = False

tokenizer = model.tokenizer 
model.max_seq_length = 4096

name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)

for idx in data:
    instance = data[idx]

    DIR_SAVE = f"./cg/{instance['instance_id']}/"
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

    codes = []
    for node in tqdm(nodes):
        if node in metadata:
            content = metadata[node].get("full", node)
            name = ":".join(node.split(":")[1:])
            if name.startswith("/"):
                name = name[1:]

            if node.startswith("FILE:") or node.startswith("CLASS:"):
                codes.append(f"{name}\n{content}")
            elif node.startswith("FUNC"):
                fname = node.strip().split(":")[-1]
                classname = fname.split(".")[0].strip() if "." in fname else None
                if classname:
                    codes.append(f"{name}\nclass {classname}:\n{content}")
                else:
                    codes.append(f"{name}\n{content}")
        else:
            name = ":".join(node.split(":")[1:])
            if name.startswith("/"):
                name = name[1:]

            codes.append(name)

    dict_codes = dict()
    for node, content in zip(nodes, codes):
        dict_codes[node] = content
    
    import json
    with open(DIR_SAVE + "dict_codes.json", "w") as f:
        json.dump(dict_codes, f, indent=4)

    emb_nodes = model.encode(nodes, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    emb_edges = model.encode([f"{h.strip()} {r.strip()} {t.strip()}" for h, r, t in edges], batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    emb_codes = model.encode(codes, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    np.save(DIR_SAVE + f"emb_nodes.npy", emb_nodes)
    np.save(DIR_SAVE + f"emb_edges.npy", emb_edges)
    np.save(DIR_SAVE + f"emb_codes.npy", emb_codes)