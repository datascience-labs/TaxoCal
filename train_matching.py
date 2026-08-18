import numpy as np
import random
import torch
import torch.nn as nn
import pickle
from models import Ours_Matching
from utils import MatchingDataset, variable_collate
from utils import metrics_hits, metrics_ndcg, metrics_mrr, metrics_auc, metrics_accuracy
from parser_s2 import args, resolve
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import os

torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = None
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

# hyperparameters
epochs = args.epoch
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay
save_path = resolve(args.ckpt_dir) if args.ckpt_dir else resolve("checkpoints")
os.makedirs(save_path, exist_ok=True)


# Load Model
# Stage-1b checkpoint. Defaults to <ckpt_dir>/{epoch}_checkpoints.pth, i.e. the original
# behaviour; pass --pretrained_ckpt to point at a specific file instead.
model_path = resolve(args.pretrained_ckpt) if args.pretrained_ckpt \
    else os.path.join(save_path, f'{epochs}_checkpoints.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Stage-1b checkpoint not found: {model_path}\n"
        f"Run 'python train_graph.py' first, or pass --pretrained_ckpt <path>."
    )
pre_trained_model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
E_s = pre_trained_model['s_embeddings']
E_j = pre_trained_model['j_embeddings']

# test data load
inference_path = resolve(args.inference_path) if args.inference_path \
    else resolve("data/inference.pkl")
if not os.path.exists(inference_path):
    raise FileNotFoundError(
        f"Matching dataset not found: {inference_path}\n"
        f"See data/README.md for how to obtain it, or pass --inference_path <path>."
    )
datas_pkl = open(inference_path,'rb')
datas = pickle.load(datas_pkl)
datasets = MatchingDataset(datas)
n_total = len(datasets)
n_train = int(n_total * 0.5)
n_valid = int(n_total * 0.1)
n_test = n_total - n_train - n_valid
trains, valids, tests = random_split(datasets, [n_train, n_valid, n_test], generator=generator)
train_loader = DataLoader(trains,  batch_size=1024, shuffle=True, collate_fn=variable_collate)
valid_loader = DataLoader(valids,  batch_size=1024, shuffle=True, collate_fn=variable_collate)
test_loader  = DataLoader(tests,   batch_size=1024, shuffle=False, collate_fn=variable_collate)

# Loss
loss_list = []
loss_cl_list = [] 
loss_reg_list = []

# Activation
activation = nn.LeakyReLU(args.activation)

model = Ours_Matching(E_j, E_s, device=device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)    


for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(tqdm(train_loader, desc="Step-2 Matching batch")):
        jids, sids = batch
        jids = jids.long().to(device)
        sids = [torch.tensor(sid, dtype=torch.long).to(device) for sid in sids]
        optimizer.zero_grad()
        loss = model(jids, sids)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().item()
        torch.cuda.empty_cache()

    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            labels, topk_preds, topk_preds5, topk_preds10, acc_preds, ncdg_mrr_preds, all_preds = list(), list(), list(), list(), list(), list(), list()
            for i, batch in enumerate(test_loader):
                jids, sids = batch
                jids = jids.long().to(device)
                sids = [torch.tensor(sid, dtype=torch.long).to(device) for sid in sids]
                label, topk_pred, topk_pred5, topk_pred10, acc_pred, ncdg_mrr_pred, all_pred = model(jids, sids, test=True)
                labels.extend(label)
                topk_preds.extend(topk_pred)
                topk_preds5.extend(topk_pred5)
                topk_preds10.extend(topk_pred10)
                acc_preds.extend(acc_pred)
                ncdg_mrr_preds.extend(ncdg_mrr_pred)
                all_preds.extend(all_pred)
            hit_rates3 = metrics_hits(labels, topk_preds)
            hit_rates5 = metrics_hits(labels, topk_preds5)
            hit_rates10 = metrics_hits(labels, topk_preds10)
            acc = metrics_accuracy(labels, acc_preds)
            ndcg3 = metrics_ndcg(labels, ncdg_mrr_preds, k=3)
            ndcg5 = metrics_ndcg(labels, ncdg_mrr_preds, k=5)
            ndcg10 = metrics_ndcg(labels, ncdg_mrr_preds, k=10)
            mrr3 = metrics_mrr(labels, ncdg_mrr_preds, k=3)
            mrr5 = metrics_mrr(labels, ncdg_mrr_preds, k=5)
            mrr10 = metrics_mrr(labels, ncdg_mrr_preds, k=10)
            auc = metrics_auc(labels, all_preds)
            print("Validation")
            print(f"HitRates@3 : {hit_rates3}, HitRates@5 : {hit_rates5}, HitRates@10 : {hit_rates10}")
            print(f"Accuracy : {acc}")
            print(f"NDCG@3 : {ndcg3}, NDCG@5 : {ndcg5}, NDCG@10 : {ndcg10}")
            print(f"MRR@3 : {mrr3}, MRR@5 : {mrr5}, MRR@10 : {mrr10}")

    if (epoch+1) % 100 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'j_embeddings': model.E_j,
            's_embeddings': model.E_s
        }, os.path.join(save_path, f"{epoch+1}_inference_checkpoints.pth"))


    print('Epoch:',epoch,'Loss:',epoch_loss)

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'j_embeddings': model.E_j,
            's_embeddings': model.E_s
        }, os.path.join(save_path, f"matching_checkpoints.pth"))

model.eval()
with torch.no_grad():
    labels, topk_preds, topk_preds5, topk_preds10, acc_preds, ncdg_mrr_preds, all_preds = list(), list(), list(), list(), list(), list(), list()
    for i, batch in enumerate(test_loader):
        jids, sids = batch
        jids = jids.long().to(device)
        sids = [torch.tensor(sid, dtype=torch.long).to(device) for sid in sids]
        label, topk_pred, topk_pred5, topk_pred10, acc_pred, ncdg_mrr_pred, all_pred = model(jids, sids, test=True)
        labels.extend(label)
        topk_preds.extend(topk_pred)
        topk_preds5.extend(topk_pred5)
        topk_preds10.extend(topk_pred10)
        acc_preds.extend(acc_pred)
        ncdg_mrr_preds.extend(ncdg_mrr_pred)
        all_preds.extend(all_pred)
    hit_rates3 = metrics_hits(labels, topk_preds)
    hit_rates5 = metrics_hits(labels, topk_preds5)
    hit_rates10 = metrics_hits(labels, topk_preds10)
    acc = metrics_accuracy(labels, acc_preds)
    ndcg3 = metrics_ndcg(labels, ncdg_mrr_preds, k=3)
    ndcg5 = metrics_ndcg(labels, ncdg_mrr_preds, k=5)
    ndcg10 = metrics_ndcg(labels, ncdg_mrr_preds, k=10)
    mrr3 = metrics_mrr(labels, ncdg_mrr_preds, k=3)
    mrr5 = metrics_mrr(labels, ncdg_mrr_preds, k=5)
    mrr10 = metrics_mrr(labels, ncdg_mrr_preds, k=10)
    auc = metrics_auc(labels, all_preds)
    print("Test")
    print(f"HitRates@3 : {hit_rates3}, HitRates@5 : {hit_rates5}, HitRates@10 : {hit_rates10}")
    print(f"Accuracy : {acc}")
    print(f"NDCG@3 : {ndcg3}, NDCG@5 : {ndcg5}, NDCG@10 : {ndcg10}")
    print(f"MRR@3 : {mrr3}, MRR@5 : {mrr5}, MRR@10 : {mrr10}")
    print(f"AUC : {auc}")