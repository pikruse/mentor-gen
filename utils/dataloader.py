import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class GraphQADataset(Dataset):
    """
    dataset that reads a test tsv `edge tests` containing questions, labels, and graph file paths.
    extracts each graph path from the desc column and caches its node features and attention masks.

    edge_test format:
    - columns: type, pos_test, question, label, desc
    - desc: filepath to an edgelist tsv (first line header, 'src\ttgt` edges)
    """

    def __init__(
            self,
            test_tsv: str,
            tokenizer: AutoTokenizer,
            max_length: int = 64,
            num_nodes: int = 50
            feature_dim: int = 128,
            ):

        df = pd.read_csv(test_tsv, sep='\t')
        self.questions = df['question'].tolist()
        self.answers = [eval(lbl)[0] for lbl in df['label']]
        self.graph_paths = df['desc'].tolist()

        # cache graph data for each unique path
        self.graph_cache = {}
        for path in set(self.graph_paths):
            if not os.path.isabs(path):
                # assume relative to test tsv directory
                root = os.path.dirname(test_tsv)
                full_path = os.path.join(root, path)
            else:
                full_path = path

            # read edges, skipping the first header line
            gdf = pd.read_csv(full_path, sep='\t', skiprows=1, names=['src', 'tgt'])

            # simplistic node features (one-hot if possible)
            if feature_dim >= num_nodes:
                node_feat = torch.eye(num_nodes, feature_dim)
            else:
                node_feat = torch.randn(num_nodes, feature_dim)

            # all nodes visible
            attn_mask = torch.ones(num_nodes, dtype=torch.long)

            self.graph_cache[path] = {
                    'node_feat': node_feat,
                    'attn_mask': attn_mask,
                    }
