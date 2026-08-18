import argparse
import csv
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Assign skill ids and emit the index files")
    parser.add_argument('--source', default='data/esco_hierarchy.csv', type=str)
    parser.add_argument('--names_out', default='data/h_skills.txt', type=str)
    parser.add_argument('--hierarchy_out', default='data/skill_hierarchy.txt', type=str)
    return parser.parse_args()


def skill_saves(results, path):
    """Write {name: index} as a "<name>\\t<index>" TSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in results.items():
            writer.writerow([key, value])
    print(f"Wrote {path} ({len(results)} skills)")


def hierarchy_saver(results, path):
    """Write each taxonomy path as a row of tab-separated ids."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for item in results:
            writer.writerow(item)
    print(f"Wrote {path} ({len(results)} paths)")


def main():
    args = parse_args()

    source = resolve(args.source)
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Hierarchy table not found: {source}\n"
            f"Run 'python preprocessing/collectors.py' first, or see data/README.md."
        )

    df = pd.read_csv(source, usecols=["level0", "level1", "level2", "level3", "level4"])
    # strip stray quotation marks from the scraped labels
    df = df.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)

    datas = df.values.tolist()
    datas_1d = [item for sublist in datas for item in sublist]

    # first-appearance ordering defines the id space
    results_index = {}
    idx = 0
    for data in datas_1d:
        if data not in results_index:
            results_index[data] = idx
            idx += 1

    results = [[results_index[d] for d in row] for row in datas]

    os.makedirs(os.path.dirname(resolve(args.names_out)), exist_ok=True)
    skill_saves(results_index, resolve(args.names_out))
    hierarchy_saver(results, resolve(args.hierarchy_out))


if __name__ == "__main__":
    main()
