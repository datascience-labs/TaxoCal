import argparse
import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the ESCO skill-taxonomy graph with BERT embeddings -> data/h_dataset.pkl")
    parser.add_argument('--model_name', default='bert-base-uncased', type=str,
                        help='HuggingFace text encoder (default: bert-base-uncased)')
    parser.add_argument('--hierarchy', default='data/skill_hierarchy.txt', type=str)
    parser.add_argument('--skills', default='data/h_skills.txt', type=str)
    parser.add_argument('--out', default='data/h_dataset.pkl', type=str)
    return parser.parse_args()


def txt2emb(model, tokenizer, texts, device):
    """Encode each text independently and return a list of (1, 768) [CLS] tensors."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Embedding"):
            encoded_input = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            output = model(**encoded_input)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu())
    return embeddings


def read_hierarchy(path):
    """Parse taxonomy paths into edges, per-node depth, and parent lists.

    Each line is one root-to-leaf path of ids. Column position *is* the depth, and each
    adjacent pair (col i, col i+1) is a parent -> child edge. A path may repeat an id in
    adjacent columns where a branch is shallower than the maximum depth; that yields a
    self-edge, which is kept so the node still aggregates its own representation.
    """
    edges, depth, parents = [], {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = [int(v) for v in line.strip().split("\t")]
            for level, node in enumerate(values):
                depth[node] = level
            for i in range(len(values) - 1):
                parent, child = values[i], values[i + 1]
                edges.append([parent, child])
                parents.setdefault(child, [])
                if parent not in parents[child]:
                    parents[child].append(parent)
    return edges, depth, parents


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hierarchy_path = resolve(args.hierarchy)
    skills_path = resolve(args.skills)
    for p in (hierarchy_path, skills_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}\nSee data/README.md.")

    edges, depth, parents = read_hierarchy(hierarchy_path)

    # The taxonomy root has no parent of its own; give it a self-edge so that it is present
    # in the adjacency and in `parents` like every other node.
    roots = [node for node, level in depth.items() if level == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one depth-0 root, found {len(roots)}: {roots}")
    root = roots[0]
    edges.append([root, root])
    parents.setdefault(root, [])
    if root not in parents[root]:
        parents[root].append(root)

    skill_names = {}
    with open(skills_path, "r", encoding="utf-8") as f:
        for line in f:
            name, idx = line.strip().split("\t")
            skill_names[int(idx)] = name

    # --- skill x skill adjacency ---
    h_rows = np.array([e[0] for e in edges])
    h_cols = np.array([e[1] for e in edges])
    h_weights = [1.0] * len(edges)
    h_size = (int(h_rows.max()) + 1, int(h_cols.max()) + 1)
    h_coo_sp = sp.coo_matrix((h_weights, (h_rows, h_cols)), shape=h_size)
    print(f"Adjacency: {h_coo_sp.shape} (skill x skill), nnz = {h_coo_sp.nnz}")
    print(f"root id = {root} | depth levels = "
          f"{ {lvl: sum(1 for v in depth.values() if v == lvl) for lvl in sorted(set(depth.values()))} }")

    # --- text embeddings, in index order ---
    print(f"Text encoder: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    names = [skill_names[i] for i in range(len(skill_names))]
    emb_skill = txt2emb(model, tokenizer, names, device)
    print(f"emb_s: {len(emb_skill)} tensors of {tuple(emb_skill[0].shape)}")

    out_path = resolve(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"arr": h_coo_sp, "emb_s": emb_skill,
                     "depth": depth, "parents": parents}, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
