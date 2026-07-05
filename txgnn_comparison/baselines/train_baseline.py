"""
Shared training/eval loop for HGT/HAN baselines.

This is the .py mirror of the logic executed in
notebooks/phase3_hgt_han_baseline.ipynb. It's kept here as the Phase 3
deliverable the plan asks for; the notebook is the runnable copy since
training actually happens on a remote Colab GPU runtime, not locally.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def sample_negatives(pos_pairs, n_drug, n_disease, rng, k=1):
    pos_set = set(map(tuple, pos_pairs))
    neg = []
    while len(neg) < len(pos_pairs) * k:
        d, s = rng.randint(0, n_drug), rng.randint(0, n_disease)
        if (d, s) not in pos_set:
            neg.append((d, s))
    return np.array(neg)


def bce_link_loss(model, z_dict, pos_pairs, n_drug, n_disease, rng, device):
    neg_pairs = sample_negatives(pos_pairs, n_drug, n_disease, rng)
    pos_idx = torch.tensor(pos_pairs, dtype=torch.long, device=device)
    neg_idx = torch.tensor(neg_pairs, dtype=torch.long, device=device)
    pos_score = model.decode(z_dict['drug'], z_dict['disease'], pos_idx[:, 0], pos_idx[:, 1])
    neg_score = model.decode(z_dict['drug'], z_dict['disease'], neg_idx[:, 0], neg_idx[:, 1])
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    return F.binary_cross_entropy_with_logits(scores, labels)


def evaluate(model, z_dict, pos_pairs, n_drug, n_disease, rng, device):
    neg_pairs = sample_negatives(pos_pairs, n_drug, n_disease, rng)
    pos_idx = torch.tensor(pos_pairs, dtype=torch.long, device=device)
    neg_idx = torch.tensor(neg_pairs, dtype=torch.long, device=device)
    with torch.no_grad():
        pos_score = torch.sigmoid(model.decode(z_dict['drug'], z_dict['disease'], pos_idx[:, 0], pos_idx[:, 1])).cpu().numpy()
        neg_score = torch.sigmoid(model.decode(z_dict['drug'], z_dict['disease'], neg_idx[:, 0], neg_idx[:, 1])).cpu().numpy()
    scores = np.concatenate([pos_score, neg_score])
    labels = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
    return {
        'AUPRC': float(average_precision_score(labels, scores)),
        'AUROC': float(roc_auc_score(labels, scores)),
    }


def train_and_eval(model, model_name, split_name, edge_index_dict, n_drug, n_disease, rng, device,
                    ind_train, ind_valid, ind_test, contra_train, contra_valid, contra_test,
                    results_dir, n_epoch=200, lr=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    param_count = sum(p.numel() for p in model.parameters())
    log = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(n_epoch):
        model.train()
        z_dict = model(edge_index_dict)
        loss_ind = bce_link_loss(model, z_dict, ind_train, n_drug, n_disease, rng, device)
        loss_contra = bce_link_loss(model, z_dict, contra_train, n_drug, n_disease, rng, device)
        loss = loss_ind + loss_contra
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append({
            'epoch': epoch,
            'loss_indication': float(loss_ind.item()),
            'loss_contraindication': float(loss_contra.item()),
        })
    train_time_min = (time.time() - t0) / 60
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == 'cuda' else None

    model.eval()
    with torch.no_grad():
        z_final = model(edge_index_dict)
    ind_metrics = evaluate(model, z_final, ind_test, n_drug, n_disease, rng, device)
    contra_metrics = evaluate(model, z_final, contra_test, n_drug, n_disease, rng, device)

    pd.DataFrame(log).to_csv(f'{results_dir}/{model_name}_{split_name}_training_log.csv', index=False)

    result = {
        'model': model_name.upper(),
        'type': 'Heterogeneous Graph Transformer (attention-based)' if model_name == 'hgt' else 'Heterogeneous Attention Network',
        'pretrained': False,
        'split': split_name,
        'indication_AUPRC': ind_metrics['AUPRC'],
        'indication_AUROC': ind_metrics['AUROC'],
        'contraindication_AUPRC': contra_metrics['AUPRC'],
        'contraindication_AUROC': contra_metrics['AUROC'],
        'training_time_min': round(train_time_min, 2),
        'param_count': param_count,
        'gpu_mem_peak_mb': round(gpu_mem_mb, 1) if gpu_mem_mb else None,
        'notes': 'Single-stage supervised training (no pretrain phase), same splits as Phase 2.',
    }
    with open(f'{results_dir}/{model_name}_{split_name}.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result
