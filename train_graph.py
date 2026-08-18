import numpy as np
import torch
import torch.nn as nn
import pickle
from models import Ours
from utils import scipy_sparse_mat_to_torch_sparse_tensor, JobSkillLoader, augmented_mean_pooling_job_emb
from parser_s1 import args, resolve
from tqdm import tqdm
import scipy.sparse as sp
import os

torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
gnn_layer = args.gnn_layer
temp = args.temp
epochs = args.epoch
dropout = args.dropout
lr = args.lr
save_path = resolve(args.ckpt_dir) if args.ckpt_dir else resolve("checkpoints")
os.makedirs(save_path, exist_ok=True)
weight_decay = args.weight_decay # L2 Regularization
lambda_1 = args.lambda1
dim = [[768, 768] for i in range(gnn_layer)]

# load data
path = resolve(args.path)
if not os.path.exists(path):
    raise FileNotFoundError(
        f"Job-skill dataset not found: {path}\n"
        f"See data/README.md for how to obtain it, or pass --path <path>."
    )
f = open(path,'rb')
datas = pickle.load(f) # arr: coo_sp, emb_j: emb_jobs, emb_s : emb_skills
train = datas['arr']
emb_j = datas['emb_j']
emb_s = datas['emb_s']

# row: job, col: skill
rowD = np.array(train.sum(1)).squeeze()  # Job degree
colD = np.array(train.sum(0)).squeeze()  # Skill degree

### ① Row-normalized: D_J^{-1} A
train_row_norm = train.copy()
for i in range(len(train_row_norm.data)):
    train_row_norm.data[i] = train_row_norm.data[i] / (rowD[train_row_norm.row[i]] + 1e-8)

### ② Col-normalized: A D_S^{-1}
train_col_norm = train.copy()
for i in range(len(train_col_norm.data)):
    train_col_norm.data[i] = train_col_norm.data[i] / (colD[train_col_norm.col[i]] + 1e-8)

### ③ Symmetric-normalized: D_J^{-1/2} A D_S^{-1/2}
train_sym_norm = train.copy()
for i in range(len(train_sym_norm.data)):
    train_sym_norm.data[i] = train_sym_norm.data[i] / (
        (rowD[train_sym_norm.row[i]] * colD[train_sym_norm.col[i]])**0.5 + 1e-8
    )

### ④ Symmetric-normalized (transposed): D_S^{-1/2} A^T D_J^{-1/2}
train_sym_reverse = sp.coo_matrix(
    (train_sym_norm.data, (train_sym_norm.col, train_sym_norm.row)),  # A^T
    shape=(train.shape[1], train.shape[0])  # skill × job
)
adj_row_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_sym_norm).coalesce().cuda(device)
adj_col_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_sym_reverse).coalesce().cuda(device)


print('Adj matrix normalized.')

# Augmentation 
# Stage-1a checkpoint. Defaults to <ckpt_dir>/{epoch}_aug_checkpoints.pth, i.e. the original
# behaviour; pass --aug_ckpt to point at a specific file instead.
model_path = resolve(args.aug_ckpt) if args.aug_ckpt \
    else os.path.join(save_path, f'{epochs}_aug_checkpoints.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Stage-1a checkpoint not found: {model_path}\n"
        f"Run 'python pretrain_taxonomy.py' first, or pass --aug_ckpt <path>."
    )
aug_e_s = torch.load(model_path, map_location=torch.device(device), weights_only=True)['embeddings']
aug_e_j = augmented_mean_pooling_job_emb(train, aug_e_s, device)

# Construct data loader
train = train.tocoo()
train_data = JobSkillLoader(train, args.num_neg_samples)
train_loader  = torch.utils.data.DataLoader(train_data, batch_size = args.batch, shuffle = True)

# Loss
loss_list = []
loss_cl_list = [] 
loss_reg_list = []

# Activation
activation = nn.LeakyReLU(args.activation)

## model params: e_j_f, e_s_f, aug_e, num_layers, dim, aug_dim, temp, activation, dropout, bias, device
model = Ours(emb_j, emb_s, aug_e_j, aug_e_s, gnn_layer, adj_row_norm, adj_col_norm, dim, temp, lambda_1, activation, dropout, bias=True, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)    


for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_loss_cl = 0
    epoch_loss_reg = 0
    epoch_loss_r = 0
    for i, batch in enumerate(tqdm(train_loader, desc="Step-1 CL batch")):
        jids, sids, pos, negs = batch
        jids = jids.long().to(device)
        sids = sids.long().to(device)
        negs = [n.long().to(device) for n in negs]

        optimizer.zero_grad()
        loss, loss_cl, loss_r, loss_reg = model(jids, sids, negs)
        # print(loss, loss_cl, loss_reg)
        # if num_gpus > 1:
        #     loss = loss.mean()
        #     loss_cl = loss_cl.mean()
        #     loss_reg = loss_reg.mean()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_cl += loss_cl.cpu().item()
        epoch_loss_reg += loss_reg.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        torch.cuda.empty_cache()


    if (epoch+1) % 100 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'j_embeddings': model.E_j,
            's_embeddings': model.E_s
        }, os.path.join(save_path, f"{epoch+1}_checkpoints.pth"))



    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_cl:',epoch_loss_cl,"Loss_r", epoch_loss_r,'Loss_reg:',epoch_loss_reg)
