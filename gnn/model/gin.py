import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

from itertools import accumulate

from .conv_layer import GINConv
from .encoder import AtomEncoder

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')



def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, output_dim, args):
        super(GraphIsomorphismNetwork, self).__init__()
        
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.readout_layer = get_readout_layers(args.readout)
        
        # graph embedding layer (GNN)
        self.atom_encoder = AtomEncoder(emb_dim = self.hidden_dim)
        
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(GINConv(self.hidden_dim))
        
        # classifier
        self.cls = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_features = self.atom_encoder(x)
        for conv in self.convs:
            node_features = conv(node_features, edge_index, edge_attr)

        # graph embedding
        for readout in self.readout_layer:
            graph_feature = readout(node_features, batch)

        logits = self.cls(graph_feature)
        probs = self.softmax(logits)

        return probs, graph_feature


def gin_train(model, optimizer, device, loader, criterion):
    model.train()
    
    acc = []
    loss_list = []
    
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        
        probs, graph_feature = model(batch)
        loss = criterion(probs, batch.y)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record
        _, prediction = torch.max(probs, -1)
        loss_list.append(loss.item())
        acc.append(prediction.eq(batch.y).cpu().numpy())
    
    return np.average(loss_list), np.concatenate(acc, axis = 0).mean()


@torch.no_grad()
def gin_evaluation(model, device, loader, criterion):
    model.eval()
    
    y = []
    loss_list = []
    pred_probs = []
    predictions = []
    
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        probs, graph_feature = model(batch)
        
        # record
        _, prediction = torch.max(probs, -1)
        y.append(batch.y)
        loss = criterion(probs, batch.y)
        
        loss_list.append(loss.item())
        predictions.append(prediction)
        pred_probs.append(probs[:, 1])
    
    loss = np.average(loss_list)
    y = torch.cat(y).cpu().numpy()
    pred_probs = torch.cat(pred_probs, dim = 0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim = 0).cpu().detach().numpy()
    
    subgraph_metric = {
        'accuracy': accuracy_score(y, predictions), 
        'precision': precision_score(y, predictions), 
        'recall': recall_score(y, predictions), 
        'f1': f1_score(y, predictions),
        'auc': roc_auc_score(y, pred_probs)
    }
    
    return loss, subgraph_metric, {}
