# Preprocessing

The datasets themselves cannot be redistributed (ESCO has its own terms; the resume corpus contains personal data). This directory contains the complete code that produced them, so the construction is fully inspectable and reproducible from the original sources.

**Text encoder: `bert-base-uncased`.** Every occupation label and every skill label — in both the job–skill graph and the skill taxonomy — is encoded independently, taking the `[CLS]` token of the last hidden state, giving a 768-d vector per label. Both graphs use the same encoder, so the two contrastive views share one embedding space; `pretrain_taxonomy.py` depends on this (its layer width is written `384*2`, i.e. 768).

## Pipeline

```text
ESCO CSV distribution
        │
        │  collectors.py            scrape level-4 concepts from the ESCO portal
        ▼
data/esco_hierarchy.csv
        │
        │  formats.py               assign integer ids, emit index files
        ▼
data/h_skills.txt + data/skill_hierarchy.txt
        │
        │  aug_graph_maker.py       bert-base-uncased  +  depth / parents
        ▼
data/h_dataset.pkl ──────────────► pretrain_taxonomy.py        (stage 1a)

data/occu-skill_graph.txt
data/occupations.txt  +  data/skills.txt
        │
        │  main_graph_pickle_makers.py    bert-base-uncased
        ▼
data/dataset.pkl ────────────────► train_graph.py  (stage 1b)
```

| Script | Produces | Notes |
|---|---|---|
| `collectors.py` | `data/esco_hierarchy.csv` | Selenium scrape, several thousand pages, hours. Only needed to rebuild the hierarchy from scratch. |
| `formats.py` | `data/h_skills.txt`, `data/skill_hierarchy.txt` | Assigns the integer skill id space used everywhere downstream. |
| `aug_graph_maker.py` | `data/h_dataset.pkl` | BERT embeddings + `depth` + `parents`. |
| `main_graph_pickle_makers.py` | `data/dataset.pkl` | BERT embeddings + job×skill adjacency. |

```bash
pip install -r preprocessing/requirements.txt

python preprocessing/collectors.py                  # optional, slow
python preprocessing/formats.py
python preprocessing/aug_graph_maker.py
python preprocessing/main_graph_pickle_makers.py

python inspect_data.py                              # confirm the result
```

Every script takes `--help`, resolves paths relative to the repository root, and can be run from any working directory.

## Output format

Run `python inspect_data.py` to print the structure of whatever you built. Against the artifacts used for the reported results it prints:

```text
dataset.pkl
  arr      coo_matrix  shape=(3039, 10644)  nnz=51229  dtype=float64
  emb_j    list  len=3039   of Tensor shape=(1, 768) dtype=torch.float32
  emb_s    list  len=10644  of Tensor shape=(1, 768) dtype=torch.float32

h_dataset.pkl
  arr      coo_matrix  shape=(10644, 10644)  nnz=42505  dtype=float64
  emb_s    list  len=10644  of Tensor shape=(1, 768) dtype=torch.float32
  depth    dict  len=10644  e.g. {10643: 0, 10391: 3, 10342: 3}
  parents  dict  len=10644  e.g. {0: [10340], 1: [10340], 2: [10340]}

inference.pkl
  jobs     dict  len=37585  e.g. {0: 1743, 1: 1743, 2: 1743}
  skills   dict  len=37585  e.g. {0: [3484], 1: [2704], 2: []}
```

Two details the training code depends on:

- **Embeddings are a *list* of `(1, 768)` tensors, not a stacked `(N, 768)` array.**
  `Ours.forward()` checks `isinstance(..., list)` and calls `torch.stack(...).squeeze(1)` on the first layer; `pretrain_taxonomy.py` does `np.array(emb_s).squeeze(1)`.
- **`depth` is positional.** `skill_hierarchy.txt` has five columns and the column index   *is* the taxonomy level. Where a branch is shallower than five levels the label repeats across adjacent columns, producing a self-edge that keeps the node aggregating its own representation.

The taxonomy is 1 root → 6 → 46 → 251 → 10,340 nodes over levels 0–4. 292 of the 10,644 skills sit under two parents; `parents` is therefore a list per skill, and only set intersections of it are used, so ordering within each list carries no meaning.

## Verification

`read_hierarchy()` in `aug_graph_maker.py` was checked against the `h_dataset.pkl` used for the reported results, starting from the shipped `skill_hierarchy.txt`:

| Property | Result |
|---|---|
| `depth` | exact match, all 10,644 entries |
| `parents` key set | exact match |
| `parents` values | set-equal for all 10,644 skills (119 differ in list order only) |
| adjacency shape / nnz | `(10644, 10644)` / `42505` — both match |
| taxonomy root | derived as `10643`, matching the value hardcoded in the original script |

## Important: skill ids are assigned, not canonical

`formats.py` assigns skill ids by **first appearance** while scanning `esco_hierarchy.csv` row by row. The id space therefore depends on the row order and row multiplicity of that CSV, and is not derived from any stable ESCO identifier.

A practical consequence, confirmed by regenerating from the current CSV:

- the label vocabulary is stable — 10,644 distinct labels either way
- but the current `esco_hierarchy.csv` has 10,559 rows, while the `skill_hierarchy.txt` used for the reported results has 10,626 (the same 10,559 distinct paths plus 67 duplicates). It is a later, de-duplicated snapshot
- so a regenerated run assigns **different integer ids to the same skills**
