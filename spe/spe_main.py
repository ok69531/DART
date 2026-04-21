import sys
sys.path.append('../')

import os
import logging
import warnings
from functools import partial
from copy import deepcopy

import numpy as np

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from modules.utils import set_seed
from modules.load_dataset import DARTDatasetREG

from spe_argument import load_spe_args
from spe_trainer import (
    create_mlp,
    create_mlp_ln,
    # get_snorm,
    calc_eigh,
    log_transform_target,
    get_param_groups,
    lr_lambda,
    training,
    evaluation
)
from spe_model import construct_model


warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


args = load_spe_args()
logging.info(args)


# args.assay_name = 'TOX21_p53_BLA_p2_ratio'
# args.tg_num = 414
# args.expose_type = 'inhale'


def main():
    args.base_hidden_dims = args.node_emb_dims
    args.phi_hidden_dims = args.node_emb_dims
    args.mlp_hidden_dims = args.node_emb_dims
    
    train_dataset = DARTDatasetREG(
        root = args.data_path, assay_name = args.assay_name, tg_num = args.tg_num, expose_type = args.expose_type,
        split = 'train', test_size = args.test_size, random_state = args.random_state
    )
    test_dataset = DARTDatasetREG(
        root = args.data_path, assay_name = args.assay_name, tg_num = args.tg_num, expose_type = args.expose_type, 
        split = 'test', test_size = args.test_size, random_state = args.random_state
    )
    train_dataset = calc_eigh(train_dataset, args)
    test_dataset = calc_eigh(test_dataset, args)
    train_dataset = log_transform_target(train_dataset)
    # test_dataset = log_transform_target(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    kwargs = {}
    kwargs['residual'] = args.residual
    kwargs['feature_type'] = 'discrete'

    if args.task == 'cls':
        criterion = nn.BCEWithLogitsLoss()
    elif args.task == 'reg':
        criterion = nn.L1Loss(reduction = 'mean')
    
    set_seed(0)
    model = construct_model(args, create_mlp, **kwargs)
    model.to(device)

    param_groups = get_param_groups(model, args)
    n_total_steps = len(train_loader) * args.n_epochs
    optimizer = optim.Adam(param_groups, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=partial(lr_lambda, args=args, n_total_steps=n_total_steps)
    )

    best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
    final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10

    early_stop = 0
    for epoch in range(1, args.n_epochs + 1):
        train_loss = training(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = evaluation(model, train_loader, criterion, device, args)
        val_mae = val_metrics['mae']; val_mse = val_metrics['mse']; val_rmse = val_metrics['rmse']; val_r2 = val_metrics['r2']
        
        if val_r2 > best_val_r2:
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            
            test_metrics = evaluation(model, test_loader, criterion, device, args, original_scale = True)
            final_test_mae = test_metrics['mae']
            final_test_mse = test_metrics['mse']
            final_test_rmse = test_metrics['rmse']
            final_test_r2 = test_metrics['r2']
            
            model_param = deepcopy(model.state_dict())
        else:
            early_stop += 1
        
        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train mae: {:.5f} | Validation mae: {:.5f}, mse: {:.5f}, rmse: {:.5f}, r2: {:.5f}'.format(train_loss, val_mae, val_mse, val_rmse, val_r2))
        
        if early_stop > 50: break

    checkpoints = {
        'params': model_param,
        'metric':{
            'test mae': final_test_mae,
            'test mse': final_test_mse,
            'test rmse': final_test_rmse,
            'test r2': final_test_r2
        }
    }
    
    save_path = f'saved_model'
    if not os.path.exists(save_path): os.makedirs(save_path)
    if args.tg_num is not None:
        torch.save(checkpoints, os.path.join(save_path, f'spe_{args.tg_num}_{args.expose_type}'))
    elif args.assay_name is not None:
        torch.save(checkpoints, os.path.join(save_path, f'spe_{args.assay_name}'))
    
    logging.info('')
    logging.info('SPE')
    if args.assay_name is not None:
        logging.info(f'Assay Name: {args.assay_name}')
    elif args.tg_num is not None:
        logging.info(f'TG {args.tg_num}-{args.expose_type}')

    logging.info('Test MAE: {:.5f}'.format(final_test_mae))
    logging.info('Test MSE: {:.5f}'.format(final_test_mse))
    logging.info('Test RMSE: {:.5f}'.format(final_test_rmse))
    logging.info('Test R2: {:.5f}'.format(final_test_r2))
    

if __name__ == '__main__':
    main()
