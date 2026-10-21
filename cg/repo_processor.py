import os
import subprocess
import tree_sitter_python as tspython
import re

from tree_sitter import Language, Parser
from file_processor import FileProcessor
from manager import Manager

PY_LANGUAGE = Language(tspython.language())
ast_parser = Parser(PY_LANGUAGE)

def is_test_file(file):
    word_list = re.split(r" |_|\/", file.lower())
    return any([word.startswith("test") for word in word_list])

class RepoProcessor:
    def __init__(self, instance, repo, base_commit):
        self.instance = instance
        self.repo = repo
        self.reponame = self.repo.replace('/', '_')
        self.repo_root = f"./_repo/{self.reponame}"
        self.base_commit = base_commit
        self.manager = Manager(self.instance)

        self.setup_repo()

    def setup_repo(self):
        url = f"https://github.com/{self.repo}.git"

        os.makedirs(self.repo_root, exist_ok=True)
        if not os.path.exists(self.repo_root + "/.git"):
            subprocess.run(["git", "clone", url, self.repo_root])
        try:
            subprocess.run(["git", "reset", "--hard", self.base_commit], cwd=self.repo_root)
        except:
            try:
                subprocess.run(["git", "fetch", "origin", self.base_commit], cwd=self.repo_root)
                subprocess.run(["git", "reset", "--hard", self.base_commit], cwd=self.repo_root)
            except:
                print(f"!!! INSTANCE {instance} FAILED !!!")
                quit()

    def collect_all_files(self):
        target_files = []
        nodes = dict()
        edges = set()
        ignore_dirs = {".git", ".github", ".venv", "env", "venv", "__pycache__", "build", "dist", "node_modules", ".idea", ".vscode"}

        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                if is_test_file(os.path.join(root, file)[len(self.repo_root):]):
                    continue
                if not (file.endswith(".py") or file.endswith(".pyi")):
                    continue

                target_files.append(os.path.join(root, file))
                    
            root = root[len(self.repo_root):]
            for d in dirs:
                parent = f"DIR:/" if len(root) == 0 else f"DIR:{root}"
                child = f"DIR:{root}/{d}"
                if is_test_file(parent) or is_test_file(child):
                    continue

                nodes[parent] = {
                    "nid": parent,
                    "type": "DIR",
                    "name": "root" if len(parent.split("/")[-1]) == 0 else parent.split("/")[-1]
                }
                nodes[child] = {
                    "nid": child,
                    "type": "DIR",
                    "name": "root" if len(child.split("/")[-1]) == 0 else child.split("/")[-1]
                }
                edges.add((parent, "CONTAINS", child))

            for f in files:
                parent = f"DIR:/" if len(root) == 0 else f"DIR:{root}"
                child = f"FILE:{root}/{f}"
                if is_test_file(parent) or is_test_file(child):
                    continue
                if not (child.endswith(".py") or child.endswith(".pyi")):
                    continue

                nodes[parent] = {
                    "nid": parent,
                    "type": "DIR",
                    "name": "root" if len(parent.split("/")[-1]) == 0 else parent.split("/")[-1]
                }
                nodes[child] = {
                    "nid": child,
                    "type": "FILE",
                    "name": "root" if len(child.split("/")[-1]) == 0 else child.split("/")[-1]
                }
                edges.add((parent, "CONTAINS", child))
        
        return target_files, nodes, edges

    def process_single_file(self, file_path, file_index):
        try:
            with open(file_path, "rb") as f:
                code_bytes = f.read()
        except Exception:
            return [], []
        
        if not (file_path.endswith(".py") or file_path.endswith(".pyi")):
            return [], []

        ast = ast_parser.parse(code_bytes)
        fileProcessor = FileProcessor(file_path, self.repo_root, file_index, ast, code_bytes)
        return fileProcessor.process_file()

    def extract_cg(self):
        all_files, nodes, edges = self.collect_all_files()
        file_index = {os.path.relpath(f, self.repo_root).replace(os.sep, "/") for f in all_files}

        from tqdm import tqdm
        for f in tqdm(all_files):
            nodes_f, edges_f = self.process_single_file(f, file_index)

            for node in nodes_f:
                if node not in nodes:
                    nodes[node] = nodes_f[node]
            edges = edges.union(edges_f)

        for src_id, r, tgt_id in edges:
            if src_id not in nodes or tgt_id not in nodes:
                print((src_id, r, tgt_id)); quit()
        
        self.manager.save(nodes, edges)
