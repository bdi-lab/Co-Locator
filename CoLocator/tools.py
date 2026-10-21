from langchain.tools import tool
import numpy as np
import copy

def create_cg_tools(cg, embedder, logger, formatter):
    @tool("search_cg", description="Search nodes in graph that are relevant to query.")
    def search_cg(query: str) -> str:
        print(f"🔍 [TOOL] search_cg({query.strip()})")
        logger.info(f"🔍 [TOOL] search_cg({query.strip()})")
        output = formatter.find_cg_node(query.strip())
        output = "\n".join(f"- {node}" for node in output)
        logger.info(output.strip())
        return output.strip()

    @tool("inspect_file", description="Provide the content of a target file.")
    def inspect_file(node: str, mode: str) -> str:
        print(f"🔍 [TOOL CALL] inspect_file({node}, {mode})")
        logger.info(f"🔍 [TOOL CALL] inspect_file({node}, {mode})")

        if not node.startswith("FILE:"):
            node = f"FILE:{node}"

        if node not in cg.metadata:
            logger.info("Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes.")
            return "Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes."

        content = cg.metadata.get(node)
        if not content:
            logger.info("Content is not available for the input node.")
            return "Content is not available for the input node."

        if mode == "full":
            content = content.get("full")
        else:
            content = content.get("middle")

        if not content:
            logger.info("Content is not available for the input node.")
            return "Content is not available for the input node."

        logger.info(content.strip())
        return content.strip()

    @tool("inspect_class", description="Provide the content of a target class.")
    def inspect_class(node: str, mode: str) -> str:
        print(f"🔍 [TOOL CALL] inspect_class({node}, {mode})")
        logger.info(f"🔍 [TOOL CALL] inspect_class({node}, {mode})")

        if not node.startswith("CLASS:"):
            node = f"CLASS:{node}"

        if node not in cg.metadata:
            logger.info("Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes.")
            return "Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes."

        content = cg.metadata.get(node)
        if not content:
            logger.info("Content is not available for the input node.")
            return "Content is not available for the input node."

        if mode == "full":
            content = content.get("full")
        else:
            content = content.get("short")
            
        if not content:
            logger.info("Content is not available for the input node.")
            return "Content is not available for the input node."

        logger.info(content.strip())
        return content.strip()

    @tool("inspect_func", description="Provide the implementation of a target function.")
    def inspect_func(node: str) -> str:
        print(f"🔍 [TOOL CALL] inspect_func({node})")
        logger.info(f"🔍 [TOOL CALL] inspect_func({node})")

        if not node.startswith("FUNC:"):
            node = f"FUNC:{node}"

        if node not in cg.metadata:
            logger.info("Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes.")
            return "Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes."

        content = cg.metadata.get(node)
        if not content:
            logger.info("Implementation is not available for the input node.")
            return "Implementation is not available for the input node."

        content = content.get("full")
        if not content:
            logger.info("Implementation is not available for the input node.")
            return "Implementation is not available for the input node."

        logger.info(content.strip())
        return content.strip()

    @tool("traverse_cg", description="Traverse Code Graph based on the given query.")
    def traverse_cg(node: str, query: str) -> str:
        print(f"🔍 [TOOL CALL] traverse_cg({node}, {query})")
        logger.info(f"🔍 [TOOL CALL] traverse_cg({node}, {query})")

        node = node.strip()
        if node.strip() not in cg.node2id:
            logger.info("Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes.")
            return "Input must match the exact representation in the Code Graph.\n[HINT] Try `search_cg` to find the exact names of nodes."

        query_embedding = embedder.encode_query(query)
        hops, beam = 2, 10

        traversal_history = [[[node]]]
        banned_edges = dict()
        for hop in range(hops):
            res = []
            for sequence in traversal_history[hop]:
                s = sequence[-1]
                edgelist = copy.deepcopy(cg.neighbor_dict[s])
                if s in banned_edges:
                    new_edgelist = []
                    for e in edgelist:
                        if e not in banned_edges[s]:
                            new_edgelist.append(e)
                    edgelist = new_edgelist

                if not edgelist:
                    continue

                emb_edges = cg.emb_edges[[idx for idx, _ in edgelist]]
                sim = np.sum(query_embedding * emb_edges, -1)
                idx_retrieve = np.argsort(-sim)[:beam]

                walks = [edgelist[i] for i in idx_retrieve]
                for i in range(len(walks)):
                    idx, (h, r, t) = walks[i]
                    if i != len(walks) - 1:
                        res.append(sequence + [False, r, t])
                    else:
                        res.append(sequence + [True, r, t])
                    if s not in banned_edges:
                        banned_edges[s] = []
                    banned_edges[s].append(walks[i])

                    if t not in banned_edges:
                        banned_edges[t] = []
                    if r.endswith("_INV"):
                        banned_edges[t].append((idx, (t, r[:-4], h)))
                    else:
                        banned_edges[t].append((idx, (t, r + "_INV", h)))

            traversal_history.append(res)

        drawings = []
        for trace in traversal_history[-1]:
            for i in range(len(trace) // 3 + 1):
                segment = trace[max(0, 3 * i - 3):3 * i + 1]
                if (i, segment) not in drawings:
                    drawings.append((i, segment))

        output = str(node)
        prefix_list = ["" for _ in range(hops + 1)]
        for depth, (_, is_last, r, t) in drawings[1:]:
            if is_last:
                prefix_list[depth] = "└─"
            else:
                prefix_list[depth] = "├─"
            prefix = "".join(prefix_list[:depth + 1])
            output += "\n" + prefix + f" {r} ── {t}"
            if is_last:
                prefix_list[depth] = "  " + " " * len(f" {r} ── ")
            else:
                prefix_list[depth] = "│ " + " " * len(f" {r} ── ")

        logger.info(f"{output.strip()}")
        return output.strip()

    return [search_cg, inspect_file, inspect_class, inspect_func, traverse_cg]


def create_kg_tools(kg, embedder, logger, formatter):
    @tool("search_kg", description="Search nodes in graph that are relevant to query.")
    def search_kg(query: str) -> str:
        print(f"🔍 [TOOL] search_kg({query.strip()})")
        logger.info(f"🔍 [TOOL] search_kg({query.strip()})")
        output = formatter.find_kg_node(query.strip())
        output = "\n".join(f"- {kg.name_map[node]}" for node in output)
        logger.info(output.strip())
        return output.strip()

    @tool("inspect_issue", description="Provide the summary of a target issue.")
    def inspect_issue(node: str) -> str:
        print(f"🔍 [TOOL CALL] inspect_issue({node})")
        logger.info(f"🔍 [TOOL CALL] inspect_issue({node})")

        node = node.upper()
        if not node.startswith("ISSUE:"):
            node = f"ISSUE:{node}"
        node = node.strip().split()[0]

        if node not in kg.metadata:
            logger.info("Input must be a valid issue in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of issues")
            return "Input must be a valid issue in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of issues."

        content = kg.metadata.get(node)
        if not content:
            logger.info("Summary is not available for the input node.")
            return "Summary is not available for the input node."

        content = content.get("text")
        if not content:
            logger.info("Summary is not available for the input node.")
            return "Summary is not available for the input node."

        logger.info(content.strip())
        return content.strip()

    @tool("inspect_pr", description="Provide the summary of a target PR.")
    def inspect_pr(node: str) -> str:
        print(f"🔍 [TOOL CALL] inspect_pr({node})")
        logger.info(f"🔍 [TOOL CALL] inspect_pr({node})")

        node = node.upper()
        if not node.startswith("PR:"):
            node = f"PR:{node}"
        node = node.strip().split()[0]

        if node not in kg.metadata:
            logger.info("Input must be a valid PR in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of PRs")
            return "Input must be a valid PR in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of PRs."

        content = kg.metadata.get(node)
        if not content:
            logger.info("Summary is not available for the input node.")
            return "Summary is not available for the input node."

        content = content.get("text")
        if not content:
            logger.info("Summary is not available for the input node.")
            return "Summary is not available for the input node."

        logger.info(content.strip())
        return content.strip()

    @tool("inspect_commit", description="Provide the summary of a target commit.")
    def inspect_commit(node: str) -> str:
        print(f"🔍 [TOOL CALL] inspect_commit({node})")
        logger.info(f"🔍 [TOOL CALL] inspect_commit({node})")

        if not node.upper().startswith("COMMIT:"):
            node = f"COMMIT:{node}"
        node = f"COMMIT:{node[7:]}"
        node = node.strip().split()[0][:13]
        
        if node not in kg.metadata:
            logger.info("Input must be a valid commit in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of commits")
            return "Input must be a valid commit in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of commits."

        content = kg.metadata.get(node)
        if not content:
            logger.info("Patch is not available for the input node.")
            return "Patch is not available for the input node."

        content = content.get("text")
        if not content:
            logger.info("Patch is not available for the input node.")
            return "Patch is not available for the input node."

        logger.info(content.strip())
        return content.strip()

    @tool("traverse_kg", description="Traverse Knowledge Graph based on the given query.")
    def traverse_kg(node: str, query: str) -> str:
        print(f"🔍 [TOOL CALL] traverse_kg({node}, {query})")
        logger.info(f"🔍 [TOOL CALL] traverse_kg({node}, {query})")

        node = node.strip()
        if node.strip() not in kg.node2id:
            logger.info("Input must match the exact representation in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of nodes.")
            return "Input must match the exact representation in the Knowledge Graph.\n[HINT] Try `search_kg` to find the exact names of nodes."

        query_embedding = embedder.encode_query(query)
        hops, beam = 2, 10

        traversal_history = [[[node]]]
        banned_edges = dict()
        for hop in range(hops):
            res = []
            for sequence in traversal_history[hop]:
                s = sequence[-1]
                edgelist = copy.deepcopy(kg.neighbor_dict[s])
                if s in banned_edges:
                    new_edgelist = []
                    for e in edgelist:
                        if e not in banned_edges[s]:
                            new_edgelist.append(e)
                    edgelist = new_edgelist

                if not edgelist:
                    continue

                emb_edges = kg.emb_edges[[idx for idx, _ in edgelist]]
                sim = np.sum(query_embedding * emb_edges, -1)
                idx_retrieve = np.argsort(-sim)[:beam]

                walks = [edgelist[i] for i in idx_retrieve]
                for i in range(len(walks)):
                    idx, (h, r, t) = walks[i]
                    if i != len(walks) - 1:
                        res.append(sequence + [False, r, t])
                    else:
                        res.append(sequence + [True, r, t])
                    if s not in banned_edges:
                        banned_edges[s] = []
                    banned_edges[s].append(walks[i])

                    if t not in banned_edges:
                        banned_edges[t] = []
                    if r.endswith("_INV"):
                        banned_edges[t].append((idx, (t, r[:-4], h)))
                    else:
                        banned_edges[t].append((idx, (t, r + "_INV", h)))

            traversal_history.append(res)

        drawings = []
        for trace in traversal_history[-1]:
            for i in range(len(trace) // 3 + 1):
                segment = trace[max(0, 3 * i - 3):3 * i + 1]
                if (i, segment) not in drawings:
                    drawings.append((i, segment))

        output = kg.name_map[str(node)]
        prefix_list = ["" for _ in range(hops + 1)]
        for depth, (_, is_last, r, t) in drawings[1:]:
            if is_last:
                prefix_list[depth] = "└─"
            else:
                prefix_list[depth] = "├─"
            prefix = "".join(prefix_list[:depth + 1])
            output += "\n" + prefix + f" {r} ── {kg.name_map[t]}"
            if is_last:
                prefix_list[depth] = "  " + " " * len(f" {r} ── ")
            else:
                prefix_list[depth] = "│ " + " " * len(f" {r} ── ")

        logger.info(f"{output.strip()}")
        return output.strip()

    return [search_kg, inspect_issue, inspect_pr, inspect_commit, traverse_kg]