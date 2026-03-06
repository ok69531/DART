import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj

from spe_mlp import MLP

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score
)


def create_mlp(in_dims: int, out_dims: int, args) -> MLP:
    return MLP(in_dims, out_dims, args, args.mlp_use_bn)


def create_mlp_ln(in_dims: int, out_dims: int, args) -> MLP:
    return MLP(in_dims, out_dims, args, args.mlp_use_ln)


# def get_snorm(instance: Data) -> Data:
#     # get the graph normalization for nodes on the fly
#     size = instance.num_nodes
#     snorm = torch.FloatTensor(size, 1).fill_(1./float(size)).sqrt()
#     instance.update({"snorm": snorm})
#     return instance


def calc_eigh(instance: Data, args) -> Data:
    # get spectrum
    n = instance.num_nodes
    L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
    L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)              # [N, N]

    Lambda = torch.zeros(1, args.pe_dims)   # [1, D_pe]
    V = torch.zeros(n, args.pe_dims)        # [N, D_pe]

    d = min(n, args.pe_dims)   # number of eigen-pairs to use (then we zero-pad up to D_pe)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)   # [N], [N, N]
    Lambda[0, :d] = eigenvalues[0:d]
    V[:, :d] = eigenvectors[:, 0:d]

    instance.update({"Lambda": Lambda, "V": V})
    
    snorm = torch.FloatTensor(n, 1).fill_(1./float(n)).sqrt()
    instance.update({"snorm": snorm})

    return instance


def get_param_groups(model, args):
    return [{
        "name": name,
        "params": [param],
        "weight_decay": 0.0 if "bias" in name else args.weight_decay
    } for name, param in model.named_parameters()]


def lr_lambda(curr_step: int, *, args, n_total_steps: int) -> float:
    if curr_step < args.n_warmup_steps:
        return float(curr_step) / float(max(1, args.n_warmup_steps))
    return max(
        0.0,
        float(n_total_steps - curr_step) / float(max(1, n_total_steps - args.n_warmup_steps))
    )


def training(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += (loss.item() * batch.y.size(0))
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluation(model, loader, criterion, device):
    model.eval()
    
    total_loss = 0
    y_true, y_pred_prob = ([] for _ in range(3))
    
    for batch in loader:
        batch = batch.to(device)
        
        score = model(batch)
        pred_prob = torch.sigmoid(score).to(torch.float32)
        
        loss = criterion(score, batch.y)
        total_loss += (loss * batch.y.size(0))
        
        y_true.append(batch.y.detach().cpu())
        y_pred_prob.append(pred_prob.detach().cpu())
    
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred_prob = torch.cat(y_pred_prob, dim = 0).numpy()
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    total_loss /= len(loader)
    
    metrics = {
        'loss': total_loss,
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_prob),
        'acc': accuracy_score(y_true, y_pred)
    }
    
    return metrics
    