import os
import posixpath
import builtins

BUILTIN_FUNCTIONS = set(dir(builtins))

import re

def normalize_symbol(text):
    text = text.replace(r'\.', '.')

    pattern = r'\([^()]*\)|\[[^[\]]*\]|\{[^{}]*\}'
    for i in range(100):
        new_text = re.sub(pattern, '', text)
        if new_text == text:
            break
        text = new_text
    
    match = re.search(r'[(\[]', text)
    if match:
        text = text[:match.start()]

    text = re.sub(r'\.{2,}', '.', text)
    text = text.strip('.')
    
    return text

class FileProcessor:
    def __init__(self, file_path, repo_root, file_index, ast, code_bytes):
        self.file_path = file_path
        self.repo_root = repo_root
        self.file_index = file_index
        self.ast = ast
        self.code_bytes = code_bytes

        self.nodes = dict()
        self.edges = set()
        self.local_symbol_table = dict()
        self.alias_map = dict()

        self.rel_path_native = os.path.relpath(file_path, repo_root)
        self.rel_path_posix = self.rel_path_native.replace(os.sep, "/")
        self.file_id = f"FILE:/{self.rel_path_posix}"
        
        self.code_lines = self.code_bytes.decode('utf-8', errors='ignore').splitlines()

    def get_node_text(self, node):
        return self.code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    def traverse_for_definitions(self, node, prev_id, parent_class=None):
        curr_id = None
        current_node_class = parent_class
        
        if node.type in ("class_definition", "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node)
                lineno = name_node.start_point[0] + 1
                
                if node.type == "class_definition":
                    prefix = "CLASS"
                    curr_id = f"CLASS:/{self.rel_path_posix}:{name}"
                    current_node_class = name 
                else:
                    prefix = "FUNC"
                    full_name = f"{parent_class}.{name}" if parent_class else name
                    curr_id = f"FUNC:/{self.rel_path_posix}:{full_name}"
                
                node_data = {
                    "nid": curr_id,
                    "type": prefix.upper(),
                    "name": name,
                    "parent_class": parent_class,
                    "lineno_start": node.start_point[0] + 1,
                    "lineno_end": node.end_point[0] + 1
                }
                
                self.local_symbol_table[name] = curr_id
                self.nodes[curr_id] = node_data
                self.edges.add((prev_id, "CONTAINS", curr_id))

        elif node.type == "assignment":
            is_global = False
            if node.parent:
                if node.parent.type == "module":
                    is_global = True
                elif node.parent.type == "expression_statement" and node.parent.parent and node.parent.parent.type == "module":
                    is_global = True

            if is_global:
                name_node = node.child_by_field_name("left")
                if name_node and name_node.type == "identifier":
                    name = self.get_node_text(name_node)
                    
                    is_constant = name.isupper()
                    is_dunder = name.startswith("__") and name.endswith("__")
                    
                    if is_constant or is_dunder:
                        lineno = name_node.start_point[0] + 1
                        curr_id = f"VAR:/{self.rel_path_posix}:{name}"
                        node_data = {
                            "nid": curr_id,
                            "type": "VAR",
                            "name": name,
                            "lineno_start": name_node.start_point[0] + 1,
                            "lineno_end": name_node.end_point[0] + 1
                        }

                        self.local_symbol_table[name] = curr_id
                        self.nodes[curr_id] = node_data
                        self.edges.add((prev_id, "CONTAINS", curr_id))
        
        next_parent = curr_id or prev_id
        for child in node.children:
            self.traverse_for_definitions(child, next_parent, current_node_class)

    def register_from_import_alias(self, module_path, imported_name, alias):
        full_import_path = f"{module_path}.{imported_name}" if module_path else imported_name
        self.alias_map[alias] = full_import_path

    def resolve_and_link_import(self, module_name, relative_level):
        current_dir = posixpath.dirname(self.rel_path_posix)
        path_parts = []
        if relative_level > 0:
            path_parts = current_dir.split('/') if current_dir != '.' else []
            if relative_level > 1:
                path_parts = path_parts[:-(relative_level - 1)]
        if module_name: path_parts.extend(module_name.split('.'))

        base_path = posixpath.join(*path_parts) if path_parts else module_name.replace('.', '/')
        candidates = [f"{base_path}.py", f"{base_path}/__init__.py"]
        for path in candidates:
            if path in self.file_index:
                self.edges.add((self.file_id, "IMPORTS", f"FILE:/{path}"))
                return

        top_pkg = module_name.split('.')[0]
        if top_pkg:
            node_data = {
                "nid": f"LIB:{top_pkg}",
                "type": "LIB",
                "name": top_pkg
            }
            self.nodes[f"LIB:{top_pkg}"] = node_data
            self.edges.add((self.file_id, "IMPORTS", f"LIB:{top_pkg}"))

    def process_imports(self):
        if not self.ast: return

        stack = [self.ast.root_node]
        while stack:
            node = stack.pop()
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        base_name = self.get_node_text(child)
                        self.resolve_and_link_import(base_name, 0)
                        self.alias_map[base_name] = base_name

                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            base_name = self.get_node_text(name_node)
                            alias = self.get_node_text(alias_node) if alias_node else base_name
                            self.resolve_and_link_import(base_name, 0)
                            self.alias_map[alias] = base_name

            elif node.type == "import_from_statement":
                module_name_node = node.child_by_field_name("module_name")
                module_path = ""
                relative_level = 0
                if module_name_node:
                    if module_name_node.type == "relative_import":
                        text = self.get_node_text(module_name_node)
                        relative_level = len(text) - len(text.lstrip('.'))
                        module_path = text.lstrip('.')
                    elif module_name_node.type == "dotted_name":
                        module_path = self.get_node_text(module_name_node)
                else:
                    for child in node.children:
                        if child.type == ".": relative_level += 1
                
                self.resolve_and_link_import(module_path, relative_level)
                
                for child in node.children:
                    if child.type == "dotted_name" and child != module_name_node:
                        imported_name = self.get_node_text(child)
                        self.register_from_import_alias(module_path, imported_name, imported_name)
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            imported_name = self.get_node_text(name_node)
                            alias = self.get_node_text(alias_node) if alias_node else imported_name
                            self.register_from_import_alias(module_path, imported_name, alias)

            for child in node.children:
                stack.append(child)

    def create_external_symbol_and_get_id(self, full_name):
        candidates = []
        for i in range(len(full_name.strip().split("."))-1):
            base_path = full_name.replace(".", "/") if i == 0 else "/".join(full_name.strip().split(".")[:-i])
            candidates += [f"{base_path}.py", f"{base_path}/__init__.py"]
        for j in range(1, len(self.rel_path_posix.strip().split("/"))-1):
            for i in range(len(full_name.strip().split("."))):
                base_path = "/".join(self.rel_path_posix.strip().split("/")[:-j])
                base_path += "/"
                base_path += full_name.replace(".", "/") if i == 0 else "/".join(full_name.strip().split(".")[:-i])
                candidates += [f"{base_path}.py", f"{base_path}/__init__.py"]
    
        for path in candidates:
            if path in self.file_index:
                return None

        top_pkg = full_name.split('.')[0]
        full_name = normalize_symbol(full_name)
        lib_id = f"LIB:{top_pkg}"
        ext_id = f"LIBSYM:{full_name}"
    
        node_data = {
            "nid": f"LIB:{top_pkg}",
            "type": "LIB",
            "name": top_pkg
        }
        self.nodes[f"LIB:{top_pkg}"] = node_data
        node_data = {
            "nid": ext_id,
            "type": "LIBSYM",
            "name": full_name
        }
        self.nodes[ext_id] = node_data
        self.edges.add((lib_id, "DEFINES", ext_id))

        return ext_id

    def resolve_symbol_and_link(self, symbol_node, source_id, edge_type):
        name = self.get_node_text(symbol_node)
        if not name or name in BUILTIN_FUNCTIONS: return
        # if not name: return
        
        target_id = None
        if name in self.local_symbol_table:
            target_id = self.local_symbol_table[name]
        else:
            prefix = name.split(".")[0]
            if prefix in self.alias_map:
                resolved_base = self.alias_map[prefix]
                suffix = '.'.join(name.split('.')[1:])
                full_name = f"{resolved_base}.{suffix}" if suffix else resolved_base
                target_id = self.create_external_symbol_and_get_id(full_name)

        if edge_type == "REFERENCES":
            if not target_id: 
                return 
            
            if target_id.startswith("VAR:"):
                pass

            elif target_id.startswith("LIBSYM:"):
                symbol_name = target_id.split(":")[-1].split(".")[-1]
                if not symbol_name.isupper():
                    return
            else:
                return

        if target_id:
            self.edges.add((source_id, edge_type, target_id))
            return

        if edge_type != "REFERENCES" and '.' not in name:
            name = normalize_symbol(name)
            unsym_id = f"UNSYM:{name}"
            node_data = {
                "nid": unsym_id,
                "type": "UNSYM",
                "name": name
            }

            self.nodes[unsym_id] = node_data
            self.edges.add((source_id, edge_type, unsym_id))

    def process_inheritance(self, class_node, class_id):
        supers_node = class_node.child_by_field_name("superclasses")
        if supers_node:
            for base_class_node in supers_node.children:
                if base_class_node.type in ["identifier", "attribute"]:
                    self.resolve_symbol_and_link(base_class_node, class_id, "INHERITS")

    def process_call(self, call_node, caller_id):
        func_node = call_node.child_by_field_name("function")
        if func_node: self.resolve_symbol_and_link(func_node, caller_id, "CALLS")
            
    def process_reference(self, ref_node, referencer_id):
        self.resolve_symbol_and_link(ref_node, referencer_id, "REFERENCES")

    def traverse_for_relations(self, node, curr_id, parent_class=None):
        current_node_class = parent_class
        
        if node.type in ("class_definition", "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node)
                lineno = name_node.start_point[0] + 1
                
                if node.type == "class_definition":
                    prefix = "CLASS"
                    curr_id = f"CLASS:/{self.rel_path_posix}:{name}"
                    current_node_class = name
                    self.process_inheritance(node, curr_id)
                else:
                    prefix = "FUNC"
                    full_name = f"{parent_class}.{name}" if parent_class else name
                    curr_id = f"FUNC:/{self.rel_path_posix}:{full_name}"
                    
        if node.type == "call":
            self.process_call(node, curr_id)
        
        if node.type == "identifier":
            parent = node.parent
            is_attribute_field = (parent and parent.type == "attribute" and parent.child_by_field_name("attribute") == node)
            is_definition = False
            if parent:
                if parent.type in ["class_definition", "function_definition", "async_function_definition"] and parent.child_by_field_name("name") == node:
                    is_definition = True
                elif parent.type == "assignment" and parent.child_by_field_name("left") == node:
                    is_definition = True
            
            is_call_target = (parent and parent.type == "call" and parent.child_by_field_name("function") == node)
            is_base_class = False
            if parent and parent.parent and parent.parent.type == "class_definition" and parent.parent.child_by_field_name("superclasses") == parent:
                is_base_class = True

            if not (is_attribute_field or is_definition or is_call_target or is_base_class):
                self.process_reference(node, curr_id)

        for child in node.children:
            self.traverse_for_relations(child, curr_id, current_node_class)

    def process_file(self):
        if self.ast == None:
            return dict(), set()

        self.traverse_for_definitions(self.ast.root_node, self.file_id, None)
        self.process_imports()
        self.traverse_for_relations(self.ast.root_node, self.file_id, None)

        return self.nodes, self.edges