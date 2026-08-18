# Data

No data files are distributed with this repository. The code that builds them is published in full under [`preprocessing/`](../preprocessing/) — see that directory's README for the pipeline. Run `python inspect_data.py` at any point to print the structure of what you have.

Expected layout after preparation:

```text
data/
├── README.md          (this file)
├── dataset.pkl        job-skill bipartite graph + BERT embeddings
├── h_dataset.pkl      ESCO skill taxonomy graph + BERT embeddings
└── inference.pkl      de-identified resume-to-job matching pairs
```


## Artifact schemas

All embeddings are produced by **`bert-base-uncased`**, taking the `[CLS]` token of the last hidden state — 768-d per label, for both the occupation/skill graph and the taxonomy.

### `dataset.pkl` — consumed by `train_graph.py` (`--path`)

```text
arr      scipy.sparse.coo_matrix  shape=(3039, 10644)  nnz=51229  dtype=float64
         rows = occupations, cols = skills, 1.0 where the occupation requires the skill
emb_j    list of 3039  torch.FloatTensor, each shape (1, 768)
emb_s    list of 10644 torch.FloatTensor, each shape (1, 768)
```

### `h_dataset.pkl` — consumed by `pretrain_taxonomy.py` (`--hpath`)

```text
arr      scipy.sparse.coo_matrix  shape=(10644, 10644)  nnz=42505  dtype=float64
         parent -> child edges of the ESCO skill taxonomy
emb_s    list of 10644 torch.FloatTensor, each shape (1, 768)
depth    dict, 10644 entries   {skill_id: level}   0 = root .. 4 = leaf
parents  dict, 10644 entries   {skill_id: [parent_id, ...]}
```

Level sizes are 1 / 6 / 46 / 251 / 10340 for levels 0-4. 292 skills have two parents.

### `inference.pkl` — consumed by `train_matching.py` (`--inference_path`)

```text
jobs     dict, 37585 entries   {user_idx: job_id}
skills   dict, 37585 entries   {user_idx: [skill_id, ...]}
```

## Source 1 — ESCO taxonomy

Occupation and skill data derive from the **ESCO** classification published by the European Commission. Download the CSV distribution from <https://esco.ec.europa.eu/en/use-esco/download> and review its terms of use before redistributing any part of it. `preprocessing/collectors.py` additionally scrapes the level-4 concepts, which are not present in the CSV distribution.

## Source 2 — resume corpus (NOT redistributed)

`inference.pkl` is derived from a public resume corpus matched against ESCO occupation and skill labels.


To rebuild `inference.pkl`, obtain that corpus from its original source under its own licence, match each resume's job title and skill strings to ESCO ids, and emit the two dictionaries above. Only integer ids should be written out.
