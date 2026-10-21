import json
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
import time

import logging
import os
import pickle
import argparse
import numpy as np

from model import Formatter, Embedder, ChatQwen, ChatGPT
from data import Graph

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
parser.add_argument("-s", type=str, default="lite")
parser.add_argument("-i", type=int, default=0)
parser.add_argument("-m", type=str, default="qwen2.5")
parser.add_argument("-limit_loc", type=int, default=60)
parser.add_argument("-limit_kg", type=int, default=10)
parser.add_argument("-seed", type=int, default=0)
parser.add_argument("-port", type=int, default=8000)
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

if len(data) <= args.i:
    raise ValueError

instance_info = data[args.i]
instance = instance_info["instance_id"]

DIR_RES = f"{name_data}/{args.m}_loc_{args.limit_loc}_kg_{args.limit_kg}_seed_{args.seed}"
if os.path.exists(f"./res/{DIR_RES}/results_{instance}.json"):
    print(f"Instance {instance} already processed!")
    quit()

def remove_duplicate(x):
    return list(dict.fromkeys(x))

### Load Graphs ###
cg = Graph(f"./../cg/cg/{instance}/")
kg = Graph(f"./../kg/kg/{instance}/")

with open(f"./../cg/cg/{instance}/dict_codes.json", "r") as f:
    dict_codes = json.load(f)
### Load Graphs ###

### Load Models ###
if args.m == "qwen2.5":
    llm = ChatOpenAI(model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.0,
        request_timeout=60,
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
        model_kwargs={"extra_body": {"repetition_penalty": 1.2}}
    )
    
    embedder = Embedder()
    formatter = Formatter(cg, kg, llm, embedder)
    chat_model = ChatQwen(llm, formatter)
elif args.m == "gpt-4o":
    llm = ChatOpenAI(model="gpt-4o-2024-08-06",
        temperature=0.0,
        request_timeout=60,
        api_key="OPENAI_API_KEY",
    )
    
    embedder = Embedder()
    formatter = Formatter(cg, kg, llm, embedder)
    chat_model = ChatGPT(llm, formatter)
### Load Models ###

### Logger Setting ###
logger = logging.getLogger(instance)
logger.setLevel(logging.INFO)

os.makedirs(f"log/{DIR_RES}", exist_ok=True)
file_handler = logging.FileHandler(f"log/{DIR_RES}/{instance}.log", encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(message)s'))

logger.handlers = []
logger.addHandler(file_handler)
logger.propagate = False
### Logger Setting ###

import re
from tqdm import tqdm

def remove_duplicate(x):
    return list(dict.fromkeys(x))

with open(f"./../CoLocator/res/{DIR_RES}/results_{instance}.json", "r") as f:
    res_raw = json.load(f)

preds_func = res_raw["refined_predictions"]
preds_str = "\n".join(f"- {fn.strip()}" for fn in preds_func)

problem_statement = instance_info.get("problem_statement")
query = f"[Problem Statement]\n{problem_statement.strip()}\n\n[Predicted Functions]\n{preds_str}"

query_embedding = embedder.encode_query(query, prefix="Given a github issue and functions to be modified, identify additional functions that need to be changed to fix the issue.")
sim = np.sum(cg.emb_codes * query_embedding, -1)
ranks = np.argsort(-sim)

pred_with_problem = []
for idx in ranks:
    if not cg.nodes[idx].startswith("FUNC:"):
        continue
    pred_with_problem.append(cg.nodes[idx])
    if len(pred_with_problem) >= 100:
        break

for fn in pred_with_problem:
    fn = ":".join(fn.split(":")[1:3])[1:]
    if fn not in preds_func:
        preds_func.append(fn)

preds_func = preds_func[:100]

prompt_system = "You are CodeRanker, an intelligent code reviewer that can analyze GitHub issues and rank code functions based on their relevance to contain the faults causing the GitHub issue."
template_user = """
I will provide you with {num} code functions, each indicated by a numerical identifier []. Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}

{candidates}

GitHub Issue: {query}. \nRank the {num} code functions above based on their relevance to contain the faults causing the GitHub issue. All the code functions should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain..
""".strip()

import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct")
eos_token = tokenizer.eos_token

def truncate_str(x, maxlen=1024):
    tokens = tokenizer.encode(x)
    return tokenizer.decode(tokens[:maxlen], skip_special_tokens=True)

problem_statement = truncate_str(problem_statement, 10240)

window_size = 10
step = 5
idx_start = len(preds_func) - window_size
while True:
    idx_end = idx_start + window_size
    candidates = preds_func[max(idx_start, 0):idx_end]

    list_cand = []
    for cand in candidates:
        body = dict_codes[f"FUNC:/{cand}"]
        truncated_body = truncate_str(body)
        list_cand.append((cand, truncated_body))

    str_candidates = "\n\n".join(f"[{idx+1}] {cand}\n{body}" for idx, (cand, body) in enumerate(list_cand))
    prompt_user = template_user.format(num=len(candidates), query=problem_statement.strip(), candidates=str_candidates)
    output = chat_model._invoke([SystemMessage(content=prompt_system), HumanMessage(content=prompt_user)]).content.strip()

    new_output = ""
    for c in output:
        if c.isdigit():
            new_output += c
        else:
            new_output += " "
    new_output = [int(x) - 1 for x in new_output.split()]
    new_output = remove_duplicate(new_output)
    print(f"Instance {args.i+1} ({idx_start}-{idx_end}):", new_output)
    replace_list = []
    for i in new_output:
        if i < 0:
            continue
        if i >= len(candidates):
            continue
        replace_list.append(candidates[i])
    replace_list = remove_duplicate(replace_list + candidates)
    
    preds_func = preds_func[:idx_start] + replace_list + preds_func[idx_end:]

    if idx_start <= 0:
        break

    idx_start -= step

total_i = chat_model.token_counts["i"] + formatter.token_i
total_o = chat_model.token_counts["o"] + formatter.token_o

print("\n" + "="*50)
print(f"📊 [TOTAL REPOSITORY ANALYSIS COST]")
print(f"📥 Total Input Tokens:  {total_i}")
print(f"📤 Total Output Tokens: {total_o}")
print(f"💰 Total Tokens Used:   {total_i + total_o}")
print("="*50)

logger.info("\n" + "="*50)
logger.info(f"📊 [TOTAL REPOSITORY ANALYSIS COST]")
logger.info(f"📥 Total Input Tokens:  {total_i}")
logger.info(f"📤 Total Output Tokens: {total_o}")
logger.info(f"💰 Total Tokens Used:   {total_i + total_o}")
logger.info("="*50)

dict_res = {
    "predictions": preds_func,
    "total_i": total_i,
    "total_o": total_o
}

os.makedirs(f"./res/{DIR_RES}", exist_ok=True)
with open(f"./res/{DIR_RES}/results_{instance}.json", "w", encoding="utf-8") as f:
    json.dump(dict_res, f, indent=4, ensure_ascii=False)