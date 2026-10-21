import re
import json
import uuid
import requests
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import AutoTokenizer
from difflib import get_close_matches

def find_closest_node(predicted_qname, all_nodes, cutoff=0.6):
    if not predicted_qname or predicted_qname in all_nodes:
        return predicted_qname
    
    matches = get_close_matches(predicted_qname, all_nodes, n=1, cutoff=cutoff)
    return matches[0] if matches else predicted_qname


PROHIBITED_TOKENS = ["```json", "}}", "<json>"]
def truncate_str(text):
    text = text.strip()
    text = text.split("<tools>")[0].strip()
    text = text.split("<|im_start|>")[0].strip()
    text = text.split("<call>")[0].strip()
    text = text.split("<tool>")[0].strip()

    truncated = True
    while truncated:
        truncated = False
        for token in PROHIBITED_TOKENS:
            if text.endswith(token):
                text = text[:-len(token)].strip()
                truncated = True

    return text

PROMPT_SYSTEM_FORMATTER = """
### Task
Identify the single most relevant node from the candidate list that matches the user's input, considering the given context.

### Rules
1. Provide the **index number** of the selected node only.
2. If no candidate node is a relevant match, you must select the index for "None of the above".
3. Strictly forbidden: Do not output any text, node names, or punctuation other than the single integer.
4. If there are multiple candidates, pick the one with the highest semantic similarity.
""".strip()

PROMPT_USER_FORMATTER = """
### Context:
{context}

### Input:
{input}

### Candidate List:
{candidates}

### Response:
""".strip()

class Formatter:
    def __init__(self, cg, kg, llm, embedder):
        self.cg = cg
        self.kg = kg
        self.llm = llm
        self.embedder = embedder

        self.token_i = 0
        self.token_o = 0

    def count_tokens(self, response):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.token_i += response.usage_metadata.get("input_tokens", 0)
            self.token_o += response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            self.token_i += usage.get("prompt_tokens", 0)
            self.token_o += usage.get("completion_tokens", 0)

    def remove_duplicate(self, x):
        return list(dict.fromkeys(x))

    def find_cg_node_instruct(self, query):
        emb_query = self.embedder.encode_query(query, prefix="Given a query, identify the most relevant nodes.")
        sim = np.sum(self.cg.emb_codes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.cg.nodes[idx] for idx in idx_retrieve]

    def find_cg_node_instruct_alternative(self, query):
        emb_query = self.embedder.encode_query(query, prefix="Given a query, identify the most relevant nodes.")
        sim = np.sum(self.cg.emb_nodes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.cg.nodes[idx] for idx in idx_retrieve]

    def find_cg_node_text(self, query):
        return get_close_matches(query, self.cg.nodes, n=8, cutoff=0.6)

    def find_cg_node(self, query):
        res_instruct = self.find_cg_node_instruct(query)
        res_alternative = self.find_cg_node_instruct_alternative(query)
        res_text = self.find_cg_node_text(query)

        return self.remove_duplicate(res_instruct + res_alternative + res_text)

    def match_cg_node_direct(self, query):
        emb_query = self.embedder.encode_query(query)
        sim = np.sum(self.cg.emb_nodes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.cg.nodes[idx] for idx in idx_retrieve]

    def match_cg_node_instruct(self, query):
        emb_query = self.embedder.encode_query(query, prefix="Given a query, identify the most relevant nodes.")
        sim = np.sum(self.cg.emb_nodes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.cg.nodes[idx] for idx in idx_retrieve]

    def match_cg_node_text(self, query):
        return get_close_matches(query, self.cg.nodes, n=8, cutoff=0.6)

    def match_cg_node(self, query, context):
        if query in self.cg.nodes:
            return query

        res_direct = self.match_cg_node_direct(query)
        res_instruct = self.match_cg_node_instruct(query)
        res_text = self.match_cg_node_text(query)

        candidates = self.remove_duplicate(res_direct + res_instruct + res_text)
        candidates.append("None of the above")
        str_candidates = "\n".join(f"[{idx+1}] {cand}" for idx, cand in enumerate(candidates))

        prompt_system = PROMPT_SYSTEM_FORMATTER
        prompt_user = PROMPT_USER_FORMATTER.format(context=context, input=query.strip(), candidates=str_candidates.strip())
        messages = [SystemMessage(content=prompt_system), HumanMessage(content=prompt_user)]

        for i in range(5):
            try:
                response = self.llm.invoke(messages)
                self.count_tokens(response)
                idx_answer = int(response.content.strip()) - 1

                if (idx_answer >= 0) and (idx_answer < len(candidates)):
                    res = candidates[idx_answer]
                    if res == "None of the above":
                        return query
                    else:
                        return res

            except:
                pass

        return query

    def find_kg_node_instruct(self, query):
        if query in self.kg.name_map:
            query = self.kg.name_map[query]
        emb_query = self.embedder.encode_query(query, prefix="Given a query, identify the most relevant nodes.")
        sim = np.sum(self.kg.emb_codes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.kg.nodes[idx] for idx in idx_retrieve]

    def find_kg_node_instruct_alternative(self, query):
        if query in self.kg.name_map:
            query = self.kg.name_map[query]
        emb_query = self.embedder.encode_query(query, prefix="Given a query, identify the most relevant nodes.")
        sim = np.sum(self.kg.emb_nodes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.kg.nodes[idx] for idx in idx_retrieve]

    def find_kg_node_text(self, query):
        return get_close_matches(query, self.kg.nodes, n=8, cutoff=0.6)

    def find_kg_node(self, query):
        res_instruct = self.find_kg_node_instruct(query)
        res_alternative = self.find_kg_node_instruct_alternative(query)
        res_text = self.find_kg_node_text(query)

        return self.remove_duplicate(res_instruct + res_alternative + res_text)

    def match_kg_node_direct(self, query):
        if query in self.kg.name_map:
            query = self.kg.name_map[query]
        emb_query = self.embedder.encode_query(query)
        sim = np.sum(self.kg.emb_nodes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.kg.nodes[idx] for idx in idx_retrieve]

    def match_kg_node_instruct(self, query):
        if query in self.kg.name_map:
            query = self.kg.name_map[query]
        emb_query = self.embedder.encode_query(query, prefix="Given a query, identify the most relevant nodes.")
        sim = np.sum(self.kg.emb_nodes * emb_query, -1)
        idx_retrieve = np.argsort(-sim)[:8]
        return [self.kg.nodes[idx] for idx in idx_retrieve]

    def match_kg_node_text(self, query):
        return get_close_matches(query, self.kg.nodes, n=8, cutoff=0.6)

    def match_kg_node(self, query, context):
        if query in self.kg.nodes:
            return query

        res_direct = self.match_kg_node_direct(query)
        res_instruct = self.match_kg_node_instruct(query)
        res_text = self.match_kg_node_text(query)

        candidates = self.remove_duplicate(res_direct + res_instruct + res_text)
        candidates.append("None of the above")
        str_candidates = "\n".join(f"- {cand}" for cand in candidates)

        prompt_system = PROMPT_SYSTEM_FORMATTER
        prompt_user = PROMPT_USER_FORMATTER.format(context=context, input=query.strip(), candidates=str_candidates.strip())
        messages = [SystemMessage(content=prompt_system), HumanMessage(content=prompt_user)]

        for i in range(5):
            try:
                response = self.llm.invoke(messages)
                self.count_tokens(response)
                idx_answer = int(response.content.strip()) - 1

                if (idx_answer >= 0) and (idx_answer < len(candidates)):
                    res = candidates[idx_answer]
                    if res == "None of the above":
                        return query
                    else:
                        return res

            except:
                pass

        return query

class Embedder:
    def __init__(self):
        self.model = "Alibaba-NLP/gte-Qwen2-7B-instruct"
        
    def encode_query(self, query, prefix=None):
        if prefix:
            input_query = f"Instruct: {prefix.strip()}\nQuery: {query.strip()}"
        else:
            input_query = query.strip()

        response = requests.post("http://localhost:9000/embed", json={"sentences": [input_query]})

        return np.array(response.json()["embeddings"][0])

class ChatQwen:
    def __init__(self, model, formatter, token_counts=None):
        self.model = model
        self.formatter = formatter

        self.max_retries = 5

        if token_counts is None:
            self.token_counts = {"i": 0, "o": 0}
        else:
            self.token_counts = token_counts

    def count_tokens(self, response):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.token_counts["i"] += response.usage_metadata.get("input_tokens", 0)
            self.token_counts["o"] += response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            self.token_counts["i"] += usage.get("prompt_tokens", 0)
            self.token_counts["o"] += usage.get("completion_tokens", 0)

    def _invoke(self, messages):
        for i in range(self.max_retries):
            try:
                response = self.model.invoke(messages)
                self.count_tokens(response)
                break
            except:
                print(f"Retry... ({i+1}/{self.max_retries})")
                if i == self.max_retries - 1:
                    quit()
        
        return response

    def invoke(self, messages, **kwargs):
        for i in range(self.max_retries):
            try:
                response = self.model.invoke(messages, stop=["}}"], **kwargs)
                self.count_tokens(response)
                break
            except:
                print(f"Retry... ({i+1}/{self.max_retries})")
                if i == self.max_retries - 1:
                    quit()

        response.content = response.content + "}}"
        if isinstance(response, AIMessage) and not response.tool_calls and response.content:
            text = response.content
            text.replace('<|im_start|>{name"', '<|im_start|>{"name"')

            json_candidates = []
            stack = 0
            start_idx = -1
            
            for i, char in enumerate(text):
                if char == '{':
                    if stack == 0:
                        start_idx = i
                    stack += 1
                elif char == '}' and stack > 0:
                    stack -= 1
                    if stack == 0 and start_idx != -1:
                        candidate_str = text[start_idx : i + 1]
                        json_candidates.append((start_idx, candidate_str))
            
            for start_pos, candidate in json_candidates:
                try:
                    tool_data = json.loads(candidate)
                    context = truncate_str(text[:start_pos].strip())

                    if "name" in tool_data:
                        response.tool_calls = [{
                            "name": tool_data["name"],
                            "args": tool_data.get("arguments", tool_data.get("args", {})),
                            "id": f"call_{uuid.uuid4().hex[:12]}"
                        }]

                        if response.tool_calls[0]["name"] in ["inspect_file", "inspect_class", "inspect_func", "traverse_cg"]:
                            target_node = response.tool_calls[0]["args"]["node"]
                            refined_node = self.formatter.match_cg_node(target_node, context)
                            response.tool_calls[0]["args"]["node"] = refined_node
                        elif response.tool_calls[0]["name"] in ["inspect_issue", "inspect_pr", "inspect_commit", "traverse_kg"]:
                            target_node = response.tool_calls[0]["args"]["node"]
                            refined_node = self.formatter.match_kg_node(target_node, context)
                            response.tool_calls[0]["args"]["node"] = refined_node
                    
                        response.content = context
                        break
                except:
                    continue
        response.content = truncate_str(response.content.strip())
        return response

    def bind_tools(self, tools, **kwargs):
        return ChatQwen(self.model.bind_tools(tools, **kwargs), self.formatter, token_counts=self.token_counts)

    def __getattr__(self, name):
        return getattr(self.model, name)

class ChatGPT:
    def __init__(self, model, formatter, token_counts=None):
        self.model = model
        self.formatter = formatter

        self.max_retries = 5

        if token_counts is None:
            self.token_counts = {"i": 0, "o": 0}
        else:
            self.token_counts = token_counts

    def count_tokens(self, response):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.token_counts["i"] += response.usage_metadata.get("input_tokens", 0)
            self.token_counts["o"] += response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            self.token_counts["i"] += usage.get("prompt_tokens", 0)
            self.token_counts["o"] += usage.get("completion_tokens", 0)

    def _invoke(self, messages):
        for i in range(self.max_retries):
            try:
                response = self.model.invoke(messages)
                self.count_tokens(response)
                break
            except:
                print(f"Retry... ({i+1}/{self.max_retries})")
                if i == self.max_retries - 1:
                    quit()
        
        return response

    def invoke(self, messages, **kwargs):
        for i in range(self.max_retries):
            try:
                response = self.model.invoke(messages, stop=["}}"], **kwargs)
                self.count_tokens(response)
                break
            except:
                print(f"Retry... ({i+1}/{self.max_retries})")
                if i == self.max_retries - 1:
                    quit()
        
        if response.tool_calls:
            if response.tool_calls[0]["name"] in ["inspect_file", "inspect_class", "inspect_func", "traverse_cg"]:
                target_node = response.tool_calls[0]["args"]["node"]
                refined_node = self.formatter.match_cg_node(target_node, response.content)
                response.tool_calls[0]["args"]["node"] = refined_node
            elif response.tool_calls[0]["name"] in ["inspect_issue", "inspect_pr", "inspect_commit", "traverse_kg"]:
                target_node = response.tool_calls[0]["args"]["node"]
                refined_node = self.formatter.match_kg_node(target_node, response.content)
                response.tool_calls[0]["args"]["node"] = refined_node

        return response

    def bind_tools(self, tools, **kwargs):
        return ChatQwen(self.model.bind_tools(tools, **kwargs), self.formatter, token_counts=self.token_counts)

    def __getattr__(self, name):
        return getattr(self.model, name)
