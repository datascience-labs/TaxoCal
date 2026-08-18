import numpy as np
import torch
import torch.nn as nn
import pickle
from models import HierarchyGNN
from utils import scipy_sparse_mat_to_torch_sparse_tensor, HierarchySkillLoader
from parser_s1 import args, resolve
from tqdm import tqdm
import os
import scipy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
gnn_layer = args.gnn_layer
temp = args.temp
epochs = args.epoch
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay # L2 Regularization
lambda_1 = args.lambda1
aug_dim = [[384*2, 384*2] for i in range(gnn_layer)]
aug_dim[-1][1] = 384*2

# load data
path = resolve(args.hpath)
if not os.path.exists(path):
    raise FileNotFoundError(
        f"Skill-taxonomy dataset not found: {path}\n"
        f"See data/README.md for how to obtain it, or pass --hpath <path>."
    )
save_path = resolve(args.ckpt_dir) if args.ckpt_dir else resolve("checkpoints")
os.makedirs(save_path, exist_ok=True)
f = open(path,'rb')
datas = pickle.load(f) # arr: coo_sp, emb_j: emb_jobs, emb_s : emb_skills
train = datas['arr']
emb_s = torch.tensor(np.array(datas['emb_s']), dtype=torch.float32, device=device).squeeze(1)
depth = datas['depth']
parents = datas['parents']
# adj coo_matrix normalized
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
train = train + scipy.sparse.eye(train.shape[0])
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')
# Construct data loader
train = train.tocoo()
train_data = HierarchySkillLoader(train, depth, parents, args.num_neg_samples)
train_loader  = torch.utils.data.DataLoader(train_data, batch_size = args.aug_batch, shuffle = True)
print('Data Loading...')

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
# # Loss
loss_list = []
loss_cl_list = [] 
loss_reg_list = []

# # Activation
activation = nn.LeakyReLU(args.activation)

# ## model params: e_j_f, e_s_f, aug_e, num_layers, dim, aug_dim, temp, activation, dropout, bias, device
model = HierarchyGNN(emb_s, args.pos_dim, aug_dim[0][0], aug_dim[0][1], aug_dim[-1][1], 5, gnn_layer, lambda_1, activation, temp, dropout, bias=True, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)

depths = list()
for idx in range(len(depth)):
    depths.append(depth[idx])
depth = torch.tensor(depths).long().to(device)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_loss_cl = 0
    epoch_loss_reg = 0
    for i, batch in enumerate(tqdm(train_loader, desc="Step-1 Aug batch")):
        cols, pos, negs, depth_batch = batch
        cols = cols.long().to(device)
        pos = pos.long().to(device)
        negs = torch.stack(negs).long().to(device)

        optimizer.zero_grad()
        loss, loss_cl, loss_r, loss_reg = model(adj_norm, cols, depth, pos, negs)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_cl += loss_cl.cpu().item()
        epoch_loss_reg += loss_reg.cpu().item()
        # epoch_loss_cl += loss_cl.cpu().item()
        # epoch_loss_reg += loss_reg.cpu().item()
        torch.cuda.empty_cache()
    if (epoch+1) % 5 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'embeddings': model.embeddings 
        }, os.path.join(save_path, f"{epoch+1}_aug_checkpoints.pth"))

    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_cl:',epoch_loss_cl,'Loss_reg:',epoch_loss_reg)