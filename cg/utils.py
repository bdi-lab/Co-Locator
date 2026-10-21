import ast
import textwrap
import inspect

def format_docstring(doc_str):
    if not doc_str: return ""
    cleaned = inspect.cleandoc(doc_str)
    if "\n" in cleaned:
        return f'"""\n{cleaned}\n"""'
    return f'"""{cleaned}"""'

def get_node_skeleton(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        doc_str = ast.get_docstring(node)
        
        orig_body = node.body
        node.body = [ast.Pass()]
        header = ast.unparse(node).split("pass")[0].rstrip()
        
        summary_content = []
        if doc_str:
            summary_content.append(format_docstring(doc_str))
        summary_content.append("...")
        
        indented_body = textwrap.indent("\n".join(summary_content), "    ")
        node.body = orig_body
        return f"{header}\n{indented_body}"
    
    return ast.unparse(node)

def summarize_file(source_code):
    try:
        tree = ast.parse(source_code)
        nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        
        result = []
        for i, node in enumerate(nodes):
            code = get_node_skeleton(node)
            if i > 0:
                prev_is_imp = isinstance(nodes[i-1], (ast.Import, ast.ImportFrom))
                curr_is_imp = isinstance(node, (ast.Import, ast.ImportFrom))
                code = ("\n" if prev_is_imp and curr_is_imp else "\n\n") + code
            result.append(code)
        return "".join(result)
    except: return ""

def summarize_class_node(class_node):
    doc_str = ast.get_docstring(class_node)
    orig_body = class_node.body
    class_node.body = [ast.Pass()]
    header = ast.unparse(class_node).split("pass")[0].rstrip()
    
    processed_members = []
    if doc_str:
        processed_members.append(format_docstring(doc_str))
        
    member_nodes = [m for m in orig_body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))]
    
    for i, m in enumerate(member_nodes):
        m_code = get_node_skeleton(m) if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) else ast.unparse(m)
        
        if i > 0 or doc_str:
            is_dense = isinstance(m, (ast.Assign, ast.AnnAssign))
            prev_m = member_nodes[i-1] if i > 0 else None
            prev_dense = isinstance(prev_m, (ast.Assign, ast.AnnAssign)) if prev_m else False
            
            sep = "\n" if is_dense and prev_dense else "\n\n"
            m_code = sep + m_code
        processed_members.append(m_code)

    class_node.body = orig_body
    return f"{header}\n{textwrap.indent(''.join(processed_members), '    ')}"

def summarize_class(class_code):
    try:
        tree = ast.parse(class_code)
        class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
        if not class_node: return ""
        return summarize_class_node(class_node)
    except: return ""

def summarize_file_with_class_details(source_code):
    try:
        tree = ast.parse(source_code)
        nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        
        result = []
        for i, node in enumerate(nodes):
            if isinstance(node, ast.ClassDef):
                code = summarize_class_node(node)
            else:
                code = get_node_skeleton(node)
                
            if i > 0:
                prev_is_imp = isinstance(nodes[i-1], (ast.Import, ast.ImportFrom))
                curr_is_imp = isinstance(node, (ast.Import, ast.ImportFrom))
                code = ("\n" if prev_is_imp and curr_is_imp else "\n\n") + code
            result.append(code)
        return "".join(result)
    except: return ""