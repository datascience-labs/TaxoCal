#!/usr/bin/env bash
# Full pipeline: taxonomy pre-training -> job-skill graph -> matching.
# Each stage picks up the previous stage's checkpoint from checkpoints/.

python pretrain_taxonomy.py
python train_graph.py
python train_matching.py
