import numpy as np
import pickle

class Graph:
    def __init__(self, path):
        self.path = path

        with open(self.path + "nodes.txt", "r") as f:
            self.nodes = [line.strip() for line in f.readlines()]
        self.node2id = {node: idx for idx, node in enumerate(self.nodes)}

        with open(self.path + "edges.txt", "r") as f:
            self.edges = [tuple(line.strip().split("\t")) for line in f.readlines()]
        self.neighbor_dict = self.construct_neighbor_dict()

        with open(self.path + "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.emb_nodes = np.load(self.path + f"emb_nodes.npy")
        self.emb_codes = np.load(self.path + f"emb_codes.npy")
        self.emb_edges = np.load(self.path + f"emb_edges.npy")

    def construct_neighbor_dict(self):
        neighbor_dict = dict()
        for idx, (h, r, t) in enumerate(self.edges):
            if h not in neighbor_dict:
                neighbor_dict[h] = []
            neighbor_dict[h].append((idx, (h, r, t)))
            if t not in neighbor_dict:
                neighbor_dict[t] = []
            neighbor_dict[t].append((idx, (t, r + "_INV", h)))

        return neighbor_dict