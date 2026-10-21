import json
from langchain.agents import create_agent
from langchain.tools import tool
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
from tools import create_cg_tools, create_kg_tools
from prompt import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
parser.add_argument("-s", type=str, default="lite")
parser.add_argument("-i", type=int, default=0)
parser.add_argument("-m", type=str, default="qwen2.5")
parser.add_argument("-limit_loc", type=int, default=60)
parser.add_argument("-limit_kg", type=int, default=10)
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

with open(f"./res_emb/{name_data}/{instance}.pkl", "rb") as f:
    preds = pickle.load(f)
    preds = [pred[1][5:] for pred in preds]

    retriever_file = remove_duplicate([pred.split(':')[0] for pred in preds])
    retriever_func = remove_duplicate([pred for pred in preds])

### Load Graphs ###
cg = Graph(f"./../cg/cg/{instance}/")
kg = Graph(f"./../kg/kg/{instance}/")

import pickle
with open(f"./../kg/kg/{instance}/name_map.pkl", "rb") as f:
    kg.name_map = pickle.load(f)
### Load Graphs ###

### Load Models ###
if args.m == "qwen2.5":
    llm = ChatOpenAI(model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.0,
        request_timeout=60,
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
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

def run_agent_loop(agent_obj, system_prompt, user_query, limit=20, agent_name="AGENT"):
    print(f"🚀 [{agent_name} STARTING] Query: {user_query[:100].strip()}...")
    logger.info(f"🚀 [{agent_name} STARTING]\nQuery: {user_query}\n")

    history = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
    all_messages = []
    config = {"recursion_limit": limit}

    try:
        for chunk in agent_obj.stream({"messages": history}, config=config, stream_mode="updates"):
            for node_name, data in chunk.items():
                if "messages" in data:
                    new_msg = data["messages"][-1]
                    all_messages.append(new_msg)
                    if node_name == "model":
                        if hasattr(new_msg, 'content') and new_msg.content:
                            text = new_msg.content if isinstance(new_msg.content, str) else str(new_msg.content)
                            print(f"\n🤔 [{agent_name} Thought]:\n{text[:100].strip()}...", flush=True)
                            logger.info(f"🤔 [{agent_name} Thought]\n{text}\n")

                    elif node_name == "tools":
                        print(f"✅ [{agent_name} Observation]: (Tool output received)", flush=True)

        response = all_messages[-1].content if all_messages else "No answer generated."
        chat_history = history + all_messages
        return response, chat_history

    except Exception as e:
        print(f"\n⚠️ [{agent_name} LIMIT REACHED/ERROR]: {str(e)}", flush=True)
        logger.info(f"⚠️ [{agent_name} LIMIT REACHED/ERROR]: {str(e)}")
        summary_prompt = (
            "You reached the maximum step limit or encountered an error. "
            "Based on the tool outputs and thoughts you've had so far, "
            "summarize your findings and provide the best possible final answer "
            "according to the requested format."
        )
        
        response = chat_model._invoke(history + all_messages + [HumanMessage(content=summary_prompt)])
        chat_history = history + all_messages + [HumanMessage(content=summary_prompt), response]

        print(f"\n📝 [{agent_name} Final Summary Generated]", flush=True)
        print(f"\n🤔 [{agent_name} Thought]:\n{response.content.strip()[:100].strip()}...", flush=True)
        logger.info(f"📝 [{agent_name} Final Summary Generated]")
        logger.info(f"🤔 [{agent_name} Thought]:\n{response.content.strip()}\n")
        return response.content, chat_history

@tool("call_kg_agent", description=(
    "Directly consult the Knowledge Graph Expert Agent. "
    "Use this tool to ask about the repository's history, past issues, pull requests, and developer intentions. "
    "Input should be a descriptive English query focusing on 'Why' or 'How' things were changed (e.g., 'Find past PRs related to auth failures'). "
    "The KG Agent will explore the knowledge graph and return a historical and contextual analysis."
))
def call_kg_agent(query: str):
    print(f"🔍 [LOC_AGENT -> KG_AGENT] {query[:100].strip()}")
    logger.info(f"🔍 [LOC_AGENT -> KG_AGENT] {query}")
    return run_agent_loop(
        kg_agent, 
        SYSTEM_PROMPT_KG.format(limit_kg=args.limit_kg//2), 
        query, 
        limit=args.limit_kg, 
        agent_name="KG_AGENT"
    )[0]

kg_tools = create_kg_tools(kg, embedder, logger, formatter)
kg_agent = create_agent(model=chat_model.bind_tools(kg_tools, parallel_tool_calls=False), tools=kg_tools)

loc_tools = create_cg_tools(cg, embedder, logger, formatter)
loc_tools.append(call_kg_agent)
loc_agent = create_agent(model=chat_model.bind_tools(loc_tools, parallel_tool_calls=False), tools=loc_tools)

problem_statement = instance_info.get("problem_statement")
query = problem_statement.strip()
query += "\n\n---"
cands = "\n".join([f"{idx}. {cand}" for idx, cand in enumerate(retriever_file[:10])])
query += f"\n\n[Suspicious Files from Retriever]\n{cands.strip()}"
query += "\n\n---"
cands = "\n".join([f"{idx}. {cand}" for idx, cand in enumerate(retriever_func[:10])])
query += f"\n\n[Suspicious Functions from Retriever]\n{cands.strip()}"

start = time.time()
final_result, overall_history = run_agent_loop(
    loc_agent,
    SYSTEM_PROMPT_LOC.format(limit_loc=args.limit_loc//2),
    query.strip(),
    limit=args.limit_loc, 
    agent_name="LOC_AGENT"
)
total_time = time.time() - start

def format_all_history(problem_statement, history, mode="all"):
    if mode == "all":
        summary = problem_statement.strip()
        summary += "\n\n---"
        cands = "\n".join([f"{idx}. {cand}" for idx, cand in enumerate(retriever_file[:10])])
        summary += f"\n\n[Suspicious Files from Retriever]\n{cands.strip()}"
        summary += "\n\n---"
        cands = "\n".join([f"{idx}. {cand}" for idx, cand in enumerate(retriever_func[:10])])
        summary += f"\n\n[Suspicious Functions from Retriever]\n{cands.strip()}"

        for msg in history[2:]:
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'content') and msg.content:
                    text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    summary += f"\n\n[LOC_AGENT]\n{text.strip()}"

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for trial in msg.tool_calls:
                        try:
                            if trial.get("name") == "kg_agent":
                                summary += f"\n\n[LOC_AGENT --> KG_AGENT]\n{trial.get('args').get('query').strip()}"
                            elif trial.get("name") == "search_cg":
                                summary += f"\n\n[TOOL CALL] search_cg({trial.get('args').get('query')})"
                            elif trial.get("name") == "traverse_cg":
                                summary += f"\n\n[TOOL CALL] traverse_cg({trial.get('args').get('node')}, {trial.get('args').get('query')})"
                            elif trial.get("name") == "inspect_file":
                                summary += f"\n\n[TOOL CALL] inspect_file({trial.get('args').get('node')}, {trial.get('args').get('mode')})"
                            elif trial.get("name") == "inspect_class":
                                summary += f"\n\n[TOOL CALL] inspect_class({trial.get('args').get('node')}, {trial.get('args').get('mode')})"
                            elif trial.get("name") == "inspect_func":
                                summary += f"\n\n[TOOL CALL] inspect_func({trial.get('args').get('node')})"
                            else:
                                summary += "\n\n[LOC_AGENT]\nINVALID TOOL CALL ATTEMPT"
                        except:
                            summary += "\n\n[LOC_AGENT]\nINVALID TOOL CALL ATTEMPT"

            if isinstance(msg, ToolMessage) and msg.name == "kg_agent":
                summary += f"\n\n[KG_AGENT --> LOC_AGENT]\n{msg.content.strip()}"

            if isinstance(msg, ToolMessage) and msg.name != "kg_agent":
                summary += f"\n{msg.content.strip()}"

            if isinstance(msg, HumanMessage):
                summary += f"\n\n[HUMAN]\n{msg.content.strip()}"

    else:
        summary = problem_statement.strip()
        summary += "\n\n---"
        cands = "\n".join([f"{idx}. {cand}" for idx, cand in enumerate(retriever_file[:10])])
        summary += f"\n\n[Suspicious Files from Retriever]\n{cands.strip()}"
        summary += "\n\n---"
        cands = "\n".join([f"{idx}. {cand}" for idx, cand in enumerate(retriever_func[:10])])
        summary += f"\n\n[Suspicious Functions from Retriever]\n{cands.strip()}"
        summary += f"\n\n[LOC_AGENT]\n{history[-1].content.strip()}"

    return summary

summary = format_all_history(problem_statement, overall_history)

format_prompt = """
Based on the interaction history, identify all functions that should be modified.
Rank them by **importance/probability**, starting with the most likely cause (the "Direct Cause" candidates first).

### Output Format (format illustrative only)
```
file: path/to/file1.py
class: MyClass1
function: my_function1

file: path/to/file2.py
class: MyClass2
function: my_function2

file: path/to/file3.py
function: my_function3
```

YOU SHOULD STRICTLY FOLLOW THE ABOVE FORMAT; JUST OUTPUT THE LOCATIONS.
""".strip()

final_result = chat_model._invoke([SystemMessage(content=format_prompt), HumanMessage(content=summary)]).content.strip()
overall_history += [SystemMessage(content=format_prompt), HumanMessage(content=summary), AIMessage(content=final_result.strip())]

os.makedirs(f"./history/{DIR_RES}", exist_ok=True)
with open(f"./history/{DIR_RES}/{instance}.pkl", "wb") as f:
    pickle.dump(overall_history, f)
final_result = final_result.replace("`", "")
print("\n" + "#"*50)
print("FINAL RESULT FROM LOCALIZATION AGENT")
print("#"*50)
print(final_result)

logger.info("#" * 50)
logger.info("FINAL RESULT FROM LOCALIZATION AGENT")
logger.info("#" * 50)
logger.info(final_result.strip())

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

import re
import json
from datetime import datetime
from difflib import get_close_matches

def find_closest_node(predicted_qname, all_nodes, cutoff=0.6):
    if not predicted_qname or predicted_qname in all_nodes:
        return predicted_qname
    
    matches = get_close_matches(predicted_qname, all_nodes, n=1, cutoff=cutoff)
    return matches[0] if matches else None

all_funcs = []
for n in cg.nodes:
    if n.startswith("FUNC:"):
        n_refine = ":".join(n.split(":")[1:])
        all_funcs.append(n_refine[1:])

lines = final_result.splitlines()
lines = [line for line in lines if line.strip()]

preds_raw, preds = [], []
current_file, current_class = None, None
for idx, line in enumerate(lines):
    line = line.strip()

    if line.lower().startswith("file:"):
        if line.lower().strip() == "file:":
            current_file = None
        else:
            current_file = line[5:].strip()
            current_class = None
    elif line.split()[0].strip().endswith(".py"):
        current_file = line.split()[0].strip()
        current_class = None
    elif line.lower().startswith("class:"):
        if line.lower().strip() == "class:":
            current_file = None
        else:
            current_class = line[6:].strip().split()[0].split("(")[0]
            if current_class.lower() in ["none", "na", "n/a"]:
                current_class = None
    elif line.startswith("module:"):
        if line.lower().strip() == "module:":
            current_file = None
        else:
            current_class = line[7:].strip().split()[0].split("(")[0]
            if current_class.lower() in ["none", "na", "n/a"]:
                current_class = None
    elif line.lower().startswith("func:") or line.startswith("function:") or line.startswith("method:"):
        if line.lower().startswith("func:"):
            if line.lower().strip() == "func:":
                fname = None
            else:
                fname = line[5:].strip().split()[0].split("(")[0]
        elif line.lower().startswith("function:"):
            if line.lower().strip() == "function:":
                fname = None
            else:
                fname = line[9:].strip().split()[0].split("(")[0]
        else:
            if line.lower().strip() == "method:":
                fname = None
            else:
                fname = line[7:].strip().split()[0].split("(")[0]
        if fname and fname.lower() in ["none", "na", "n/a"]:
            fname = None

        if fname:
            if "." in fname:
                q_name = f"{current_file}:{fname}"
            elif current_class:
                q_name = f"{current_file}:{current_class}.{fname}"
            else:
                q_name = f"{current_file}:{fname}"
            preds_raw.append(q_name)
            refined_q_name = find_closest_node(q_name, all_funcs)
            if refined_q_name:
                preds.append(refined_q_name)

def remove_duplicate(x):
    return list(dict.fromkeys(x))

dict_res = {
    "raw_text": final_result,
    "predictions": remove_duplicate(preds_raw),
    "refined_predictions": remove_duplicate(preds),
    "total_time": total_time,
    "total_i": total_i,
    "total_o": total_o
}

os.makedirs(f"./res/{DIR_RES}", exist_ok=True)
with open(f"./res/{DIR_RES}/results_{instance}.json", "w", encoding="utf-8") as f:
    json.dump(dict_res, f, indent=4, ensure_ascii=False)

print("\n" + "#" * 50)
print("REFINED RESULT FROM LOCALIZATION AGENT")
print("#" * 50)
print("\n".join([f"- {fn}" for fn in remove_duplicate(preds)]))

logger.info("#" * 50)
logger.info("REFINED RESULT FROM LOCALIZATION AGENT")
logger.info("#" * 50)
logger.info("\n".join([f"- {fn}" for fn in remove_duplicate(preds)]))