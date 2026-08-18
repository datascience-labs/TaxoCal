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
    parser = argparse.ArgumentParser(description="Build the job-skill graph with BERT embeddings -> data/dataset.pkl")
    parser.add_argument('--model_name', default='bert-base-uncased', type=str,
                        help='HuggingFace text encoder (default: bert-base-uncased)')
    parser.add_argument('--graph', default='data/occu-skill_graph.txt', type=str)
    parser.add_argument('--occupations', default='data/occupations.txt', type=str)
    parser.add_argument('--skills', default='data/skills.txt', type=str)
    parser.add_argument('--out', default='data/dataset.pkl', type=str)
    return parser.parse_args()


def load_index_file(path):
    """Read a "<name>\\t<index>" TSV into {index: name}."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name, idx = line.strip().split("\t")
            mapping[int(idx)] = name
    return mapping


def txt2emb(model, tokenizer, texts, device):
    """Encode each text independently and return a list of (1, 768) [CLS] tensors."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Embedding"):
            encoded_input = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            output = model(**encoded_input)
            # [CLS] token of the last hidden state -> (1, hidden_size)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu())
    return embeddings


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Text encoder: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    graph_path = resolve(args.graph)
    occ_path = resolve(args.occupations)
    skill_path = resolve(args.skills)
    for p in (graph_path, occ_path, skill_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}\nSee data/README.md.")

    # --- edges ---
    graphs = []
    with open(graph_path, "r", encoding="utf-8") as f:
        for line in f:
            graphs.append(list(map(int, line.strip().split("\t"))))

    idx_jobs = load_index_file(occ_path)
    idx_skills = load_index_file(skill_path)

    # --- job x skill adjacency ---
    rows, cols, weights = [], [], []
    for start, dest in graphs:
        rows.append(start)
        cols.append(dest)
        weights.append(1.0)
    size = (len(idx_jobs), len(idx_skills))
    coo_sp = sp.coo_matrix((weights, (np.array(rows), np.array(cols))), shape=size)
    print(f"Adjacency: {coo_sp.shape} (job x skill), nnz = {coo_sp.nnz}")

    # --- text embeddings, in index order ---
    job_names = [idx_jobs[i] for i in range(len(idx_jobs))]
    skill_names = [idx_skills[i] for i in range(len(idx_skills))]
    emb_jobs = txt2emb(model, tokenizer, job_names, device)
    emb_skills = txt2emb(model, tokenizer, skill_names, device)
    print(f"emb_j: {len(emb_jobs)} tensors of {tuple(emb_jobs[0].shape)}")
    print(f"emb_s: {len(emb_skills)} tensors of {tuple(emb_skills[0].shape)}")

    out_path = resolve(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"arr": coo_sp, "emb_j": emb_jobs, "emb_s": emb_skills}, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
