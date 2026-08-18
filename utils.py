import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

random.seed(4321)


def sparse_transpose(sparse_mat):
    sparse_mat = sparse_mat.coalesce()
    indices = sparse_mat.indices()
    values = sparse_mat.values()
    shape = sparse_mat.shape
    return torch.sparse_coo_tensor(indices[[1, 0], :], values, (shape[1], shape[0])).coalesce().to(sparse_mat.device)

def metrics_hits(labels, topk_preds):
    hits = [label in pred for label, pred in zip(labels, topk_preds)]
    hit_rate = sum(hits) / len(labels)
    return hit_rate
 

def metrics_ndcg(labels, predictions, k = 5):
    def ndcg_at_k(true_item, predicted_items, k=5):
        if true_item in predicted_items[:k]:
            rank = predicted_items.tolist().index(true_item) + 1  # 1-based index
            dcg = 1 / math.log2(rank + 1)
        else:
            dcg = 0.0
        idcg = 1.0
        ndcg = dcg / idcg
        return ndcg
    ndcgs = [ndcg_at_k(t, p, k) for t, p in zip(labels, predictions)]
    return sum(ndcgs) / len(ndcgs)


def metrics_mrr(labels, predictions, k = 5):
    def mrr_at_k(true_item, predicted_items, k):
        if true_item in predicted_items[:k]:
            rank = predicted_items.tolist().index(true_item) + 1  # 1-based
            return 1 / rank
        return 0.0
    reciprocal_ranks = [mrr_at_k(label, pred, k) for label, pred in zip(labels, predictions)]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def metrics_auc(labels, predictions):
    aucs = list()
    for label, predict in zip(labels, predictions):
        y_true = [0.0 for i in range(len(predict))]
        y_true[label] = 1.0
        predict = predict.cpu().numpy()
        auc = roc_auc_score(y_true, predict)
        aucs.append(auc)
    return sum(aucs) / len(aucs) if aucs else 0.0

def metrics_accuracy(y_true, y_pred):
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    accuracy = correct / len(y_true)
    return accuracy


def sparse_dropout(mat, dropout, device):
    mat = mat.coalesce()
    indices = mat.indices()
    values = mat.values()

    random_tensor = torch.rand(values.shape, device=device) + (1 - dropout)
    dropout_mask = random_tensor.floor().bool()

    indices = indices[:, dropout_mask]
    values = values[dropout_mask] / (1 - dropout)

    return torch.sparse_coo_tensor(indices, values, mat.shape).to(device)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def augmented_mean_pooling_job_emb(adj, aug_e_s, device):
    rows, cols = adj.row, adj.col
    aug_job_emb = list()

    for idx in range(max(rows)+1):
        neighbors = cols[rows == idx]
        embs = aug_e_s[neighbors]
        mean_emb = embs.mean(dim=0)
        aug_job_emb.append(mean_emb)
    stacked = torch.stack([t.to(device) for t in aug_job_emb])
    return stacked

def dot_product_scipy(mat_sparse, mat_dense):
    return torch.sparse.mm(mat_sparse, mat_dense.squeeze(1))



def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

class JobSkillLoader(data.Dataset):
    def __init__(self, coo_matrix, num_neg_samples = 1, hard_negative=False):
        self.rows = coo_matrix.row
        self.cols = coo_matrix.col
        self.num_neg_samples = num_neg_samples
        self.hard_negative = hard_negative
        self.d_mat = coo_matrix.todok()
        if not hard_negative:
            self.skills_mapper = self.build_skill_job_mapping()
            self.pos, self.negs = self.generate_skill_pairs()
        else:
            self.num_skills = coo_matrix.shape[1]
            self.num_jobs = coo_matrix.shape[0]
            self.skill_job_matrix = self.build_skill_job_matrix()
            self.pos, self.negs = self.generate_hard_negative_pairs()

    def build_skill_job_matrix(self):
        """ skill x job sparse matrix (transpose of job-skill COO) """
        return coo_matrix(
            (np.ones_like(self.cols), (self.cols, self.rows)),  # (skill, job)
            shape=(self.num_skills, self.num_jobs)
        ).tocsr()


    def build_skill_job_mapping(self):
        """  {job_id: set(skill_id)} make mapping dict using COO Matrix"""
        skill_jobs = {}
        for job, skill in zip(self.rows, self.cols):
            if skill not in skill_jobs:
                skill_jobs[skill] = set()
            skill_jobs[skill].add(job)
        return skill_jobs


    def generate_skill_pairs(self):
        positive_pairs = []
        negative_skill_pairs = []
        
        skill_list = list(self.skills_mapper.keys())  # skill list
        
        # Positive Pairs 
        while len(positive_pairs) < len(self.rows):
            for skill_a in skill_list:
                for skill_b in skill_list:
                    if skill_a != skill_b:
                        common_jobs = self.skills_mapper[skill_a] & self.skills_mapper[skill_b]
                        if common_jobs:
                            positive_pairs.append(skill_b)

        # Negative Pairs 
        while len(negative_skill_pairs) < len(self.rows):
            for skill_a in skill_list:
                if len(negative_skill_pairs) >= len(self.rows): break
                temp_list = set()
                while len(temp_list) < self.num_neg_samples:
                    skill_b = random.choice(skill_list)
                    if skill_a != skill_b and not (self.skills_mapper[skill_a] & self.skills_mapper[skill_b]):
                        temp_list.add(skill_b)
                negative_skill_pairs.append(list(temp_list))
        return positive_pairs, negative_skill_pairs
    
    def generate_hard_negative_pairs(self):
        skill_vectors = self.skill_job_matrix.toarray()  # (num_skills, num_jobs)
        similarities = cosine_similarity(skill_vectors)  # (num_skills, num_skills)

        pos_pairs = []
        hard_negs = []

        for job, skill_a in zip(self.rows, self.cols):
            sim_vec = similarities[skill_a]
            sorted_idx = np.argsort(-sim_vec)

            # Positive
            pos_found = False
            for idx in sorted_idx:
                if idx != skill_a and self.d_mat.get((job, idx)) == 1:
                    pos_pairs.append(idx)
                    pos_found = True
                    break
            if not pos_found:
                pos_pairs.append(skill_a)  # fallback

            # Hard Negative
            neg_list = []
            for idx in sorted_idx:
                if len(neg_list) >= self.num_neg_samples:
                    break
                job = int(job)
                skill_a = int(skill_a)
                idx = int(idx)

                if (
                    0 <= job < self.num_jobs and
                    0 <= idx < self.num_skills and
                    idx != skill_a and
                    self.d_mat.get((job, idx)) != 1
                ):
                    neg_list.append(idx)

            # fallback 
            while len(neg_list) < self.num_neg_samples:
                fallback = random.randint(0, self.num_skills - 1)
                if fallback != skill_a:
                    neg_list.append(fallback)

            hard_negs.append(neg_list)

        while len(pos_pairs) < len(self.rows):
            pos_pairs.append(random.randint(0, self.num_skills - 1))
        while len(hard_negs) < len(self.rows):
            negs = random.sample(range(self.num_skills), self.num_neg_samples)
            hard_negs.append(negs)

        return pos_pairs, hard_negs

                
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.pos[idx],self.negs[idx]


class HierarchySkillLoader(data.Dataset):
    def __init__(self, coo_matrix, depth, parents, num_neg_samples = 1,  hard_negative=False):
        self.rows = coo_matrix.row
        self.cols = coo_matrix.col
        self.num_pos_samples, self.num_neg_samples = 1, num_neg_samples
        self.d_mat = coo_matrix.todok()
        self.parnets = parents
        self.depth = depth
        if not hard_negative:
            self.pos, self.negs = self.generate_skill_pairs()
        else:
            self.skill_list = list(self.parnets.keys())
            self.depth_dict = {idx: self.depth[idx] for idx in self.skill_list}
            self.pos, self.negs = self.generate_hard_negative_pairs()
    
    # random child를 select
    def generate_skill_pairs(self):
        positive_pairs, negative_pairs = [], []
        skill_list = list(self.parnets.keys())

        depth_dict = {idx: self.depth[idx] for idx in skill_list}

        # POSITIVE SAMPLING 
        while len(positive_pairs) < len(self.rows):
            for skill_a in skill_list:
                if len(positive_pairs) >= len(self.rows): break
                pos_samples = 0
                while pos_samples < self.num_pos_samples:
                    skill_b = random.choice(skill_list)
                    if (
                        skill_a != skill_b and
                        set(self.parnets[skill_a]) & set(self.parnets[skill_b]) and
                        abs(depth_dict[skill_a] - depth_dict[skill_b]) <= 1
                    ):
                        positive_pairs.append(skill_b)
                        pos_samples += 1
                    elif skill_a == 5605 or skill_a == 10643:  # exception
                        positive_pairs.append(skill_a)
                        pos_samples += 1

        # NEGATIVE SAMPLING 
        while len(negative_pairs) < len(self.rows):
            for skill_a in skill_list:
                if len(negative_pairs) >= len(self.rows): break
                neg_samples = []
                while len(neg_samples) < self.num_neg_samples:
                    skill_b = random.choice(skill_list)
                    if (
                        skill_b != 10423 and
                        skill_a != skill_b and
                        skill_b not in neg_samples and
                        not set(self.parnets[skill_a]) & set(self.parnets[skill_b]) and
                        abs(depth_dict[skill_a] - depth_dict[skill_b]) <= 1
                    ):
                        neg_samples.append(skill_b)
                    elif (
                        skill_a != skill_b and
                        skill_b not in neg_samples and
                        not set(self.parnets[skill_a]) & set(self.parnets[skill_b])
                    ):
                        neg_samples.append(skill_b)
                random.shuffle(neg_samples)
                negative_pairs.append(neg_samples)

        return positive_pairs, negative_pairs
    def generate_hard_negative_pairs(self):
        positive_pairs, hard_negative_pairs = [], []

        for skill_a in self.cols:
            skill_a = int(skill_a)

            # Positive
            pos_candidates = [
                skill_b for skill_b in self.skill_list
                if skill_a != skill_b and
                set(self.parnets[skill_a]) & set(self.parnets[skill_b]) and
                abs(self.depth_dict[skill_a] - self.depth_dict[skill_b]) <= 1
            ]
            positive_pairs.append(random.choice(pos_candidates) if pos_candidates else skill_a)

            # Hard Negative
            neg_candidates = [
                skill_b for skill_b in self.skill_list
                if skill_a != skill_b and
                not set(self.parnets[skill_a]) & set(self.parnets[skill_b]) and
                abs(self.depth_dict[skill_a] - self.depth_dict[skill_b]) <= 1
            ]
            random.shuffle(neg_candidates)
            if len(neg_candidates) >= self.num_neg_samples:
                hard_negative_pairs.append(neg_candidates[:self.num_neg_samples])
            else:
                fallback = random.sample(self.skill_list, self.num_neg_samples - len(neg_candidates))
                hard_negative_pairs.append(neg_candidates + fallback)

        while len(positive_pairs) < len(self.rows):
            positive_pairs.append(random.choice(self.skill_list))
        while len(hard_negative_pairs) < len(self.rows):
            hard_negative_pairs.append(random.sample(self.skill_list, self.num_neg_samples))

        return positive_pairs, hard_negative_pairs

    def __len__(self):
        return len(self.depth)
    
    def __getitem__(self, idx):
        return self.cols[idx], self.pos[idx], self.negs[idx], self.depth[idx]
    
def variable_collate(batch):
    """
    batch: list of tuples (job_id: int, skill_list: List[int], label: float)
    returns:
      job_ids: LongTensor [B]
      skill_lists: List[List[int]]
      labels: FloatTensor [B]
    """
    job_ids     = torch.tensor([item[0] for item in batch], dtype=torch.long)
    skill_lists = [item[1]              for item in batch]   # variable-length
    return job_ids, skill_lists




class MatchingDataset(data.Dataset):
    """
    samples = [(job_id, skill_id, label), ...]
    label = 1  (positive), 0 (sampled negative)
    """
    def __init__(self, datas):
        self.jobs = datas['jobs']
        self.skills = datas['skills']
    def __len__(self): 
        return len(self.jobs)
    
    def __getitem__(self, idx):
        return self.jobs[idx], self.skills[idx]
