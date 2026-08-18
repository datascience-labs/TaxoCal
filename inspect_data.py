import argparse
import itertools
import os
import pickle

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPECTED = {
    "dataset.pkl": """  arr    scipy.sparse.coo_matrix (n_jobs, n_skills), 1.0 where job requires skill
  emb_j  list of n_jobs   torch.FloatTensor, each (1, 768)   bert-base-uncased [CLS]
  emb_s  list of n_skills torch.FloatTensor, each (1, 768)   bert-base-uncased [CLS]""",
    "h_dataset.pkl": """  arr      scipy.sparse.coo_matrix (n_skills, n_skills), parent -> child
  emb_s    list of n_skills torch.FloatTensor, each (1, 768)  bert-base-uncased [CLS]
  depth    {skill_id: level}, 0 = taxonomy root .. 4 = leaf
  parents  {skill_id: [parent_id, ...]}, a skill may sit under several branches""",
    "inference.pkl": """  jobs    {user_idx: job_id}          ground-truth occupation per user
  skills  {user_idx: [skill_id, ...]}  skills extracted from that user's resume
  Integer ids only -- contains no personal data.""",
}


def describe(value, indent="    "):
    import numpy as np

    kind = type(value).__name__
    if hasattr(value, "shape") and hasattr(value, "nnz"):          # scipy sparse
        return f"{kind}  shape={value.shape}  nnz={value.nnz}  dtype={value.dtype}"
    if hasattr(value, "shape") and not isinstance(value, (list, dict)):   # tensor / ndarray
        return f"{kind}  shape={tuple(value.shape)}  dtype={value.dtype}"
    if isinstance(value, list):
        if not value:
            return f"list  len=0"
        e = value[0]
        inner = (f"{type(e).__name__} shape={tuple(e.shape)} dtype={e.dtype}"
                 if hasattr(e, "shape") else type(e).__name__)
        return f"list  len={len(value)}  of {inner}"
    if isinstance(value, dict):
        items = list(itertools.islice(value.items(), 3))
        sample = ", ".join(f"{k!r}: {v!r}" for k, v in items)
        if len(sample) > 90:
            sample = sample[:90] + "..."
        return f"dict  len={len(value)}  e.g. {{{sample}}}"
    return kind


def inspect(path):
    name = os.path.basename(path)
    print("=" * 78)
    print(name)
    print("=" * 78)
    if not os.path.exists(path):
        print(f"  not found: {path}")
        if name in EXPECTED:
            print("  expected structure:")
            print(EXPECTED[name])
        print()
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  file size : {size_mb:.1f} MB")
    print(f"  top level : {type(data).__name__} with keys {list(data.keys())}")
    for key, value in data.items():
        print(f"    {key:<9}{describe(value)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print the on-disk structure of the preprocessed datasets")
    parser.add_argument('--path', default=None, type=str,
                        help='inspect a single pickle instead of all three')
    parser.add_argument('--data_dir', default='data', type=str)
    args = parser.parse_args()

    if args.path:
        p = args.path if os.path.isabs(args.path) else os.path.join(ROOT, args.path)
        inspect(p)
        return

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)
    for name in ("dataset.pkl", "h_dataset.pkl", "inference.pkl"):
        inspect(os.path.join(data_dir, name))


if __name__ == "__main__":
    main()
