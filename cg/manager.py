import os
import pickle

from tqdm import tqdm
from utils import *

class Manager:
    def __init__(self, instance):
        self.instance = instance
        self.repo = self.instance["repo"]

    def normalize_name(self, nid):
        return "".join(nid.strip().split()).strip()

    def extract_lines(self, repo, file, start=0, end=0):
        path = f"./_repo/{repo.replace('/', '_')}{file}"
        try:
            with open(path, "r") as f:
                lines = [line.rstrip() for line in f.readlines()]

            if start == 0 and end == 0:
                return "\n".join(lines)
                
            s_idx = max(0, start - 1)
            e_idx = min(len(lines), end)
            return "\n".join(lines[s_idx:e_idx])
        except:
            return ""

    def extract_content(self, node_data):
        if node_data["type"] not in ("FILE", "CLASS", "FUNC"):
            return None

        if node_data["type"] == "FILE":
            file = node_data['nid'].strip().split(":")[1]
            full = self.extract_lines(self.repo, file)
            try:
                middle = summarize_file_with_class_details(full)
            except:
                middle = full
            try:
                short = summarize_file(full)
            except:
                short = full
            return {"full": full, "middle": middle, "short": short}

        if node_data["type"] == "CLASS":
            file = node_data['nid'].strip().split(":")[1]
            full = self.extract_lines(self.repo, file, int(node_data['lineno_start']), int(node_data['lineno_end']))
            try:
                short = summarize_class(full)
            except:
                short = full
            return {"full": full, "short": short}

        if node_data["type"] == "FUNC":
            file = node_data['nid'].strip().split(":")[1]
            full = self.extract_lines(self.repo, file, int(node_data['lineno_start']), int(node_data['lineno_end']))
            return {"full": full}

    def save(self, nodes, edges):
        nodes = [nodes[n] for n in nodes]
        for n in nodes:
            n["nid"] = self.normalize_name(n["nid"])
        nodes = sorted(nodes, key=lambda x: x["nid"])
        edges = sorted((self.normalize_name(src_id), r.strip(), self.normalize_name(tgt_id)) for (src_id, r, tgt_id) in edges)

        metadata = dict()
        for n in tqdm(nodes):
            content = self.extract_content(n)
            if content:
                metadata[n["nid"]] = content

        DIR_SAVE = f"./cg/{self.instance['instance_id']}/"
        os.makedirs(DIR_SAVE, exist_ok=True)

        with open(DIR_SAVE + "nodes.txt", "w") as f:
            for node in nodes:
                f.write(node["nid"] + "\n")

        with open(DIR_SAVE + "edges.txt", "w") as f:
            for src_id, edge_type, tgt_id in edges:
                f.write(f"{src_id}\t{edge_type}\t{tgt_id}\n")

        with open(DIR_SAVE + "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)