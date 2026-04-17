import sys
sys.path.append('../')

import wandb
import logging
import warnings
from functools import partial

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


device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


args = load_spe_args()
logging.info(args)


wandb.login(key = open('../wandb_key.txt', 'r').readline())
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'avg val f1'},
    'parameters':{
        'n_epochs': {'values': [300, 500, 1000]},
        'lr': {'values': [0.005, 0.001]},
        'weight_decay': {'values': [0, 1e-5, 3e-6]},
        
        'node_emb_dims': {'values': [32, 64, 128]},
        'n_phi_layers': {'values': [2, 4, 8]},
        'n_psi_layers': {'values': [2, 3, 4]},
        'n_base_layers': {'values': [2, 3, 4]},
        'n_mlp_layers': {'values': [2, 3, 4]},
    }
}
sweep_id = wandb.sweep(sweep_configuration, project = f'DART-REG-SPE')

# args.assay_name = 'TOX21_p53_BLA_p2_ratio'
# args.tg_num = 414
# args.expose_type = 'inhale'

def main():
    wandb.init()
    
    args.n_epochs = wandb.config.n_epochs
    args.lr = wandb.config.lr
    args.weight_decay = wandb.config.weight_decay
    args.node_emb_dims = wandb.config.node_emb_dims
    args.n_phi_layers = wandb.config.n_phi_layers
    args.n_psi_layers = wandb.config.n_psi_layers
    args.n_base_layers = wandb.config.n_base_layers
    args.n_mlp_layers = wandb.config.n_mlp_layers
    
    args.base_hidden_dims = args.node_emb_dims
    args.phi_hidden_dims = args.node_emb_dims
    args.mlp_hidden_dims = args.node_emb_dims
    
    if args.tg_num is not None:
        wandb.run.name = f'tg{args.tg_num}-{args.expose_type}'
    elif args.assay_name is not None:
        wandb.run.name = f'{args.assay_name}'
    
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
    test_dataset = log_transform_target(test_dataset)

    kwargs = {}
    kwargs['residual'] = args.residual
    kwargs['feature_type'] = 'discrete'

    val_mae_list, val_mse_list, val_rmse_list, val_r2_list = ([] for _ in range(4))
    for seed in range(3):
        set_seed(seed)

        num_train = int(len(train_dataset) * 0.8)
        num_val = len(train_dataset) - num_train
        train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True)
        val_loader = DataLoader(val, batch_size = args.batch_size, shuffle = False)
        # test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
        
        if args.task == 'cls':
            criterion = nn.BCEWithLogitsLoss()
        elif args.task == 'reg':
            criterion = nn.L1Loss(reduction = 'mean')
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
            val_metrics = evaluation(model, val_loader, criterion, device, args)
            val_mae = val_metrics['mae']; val_mse = val_metrics['mse']; val_rmse = val_metrics['rmse']; val_r2 = val_metrics['r2']
            
            if val_r2 > best_val_r2:
                best_val_mae = val_mae
                best_val_mse = val_mse
                best_val_rmse = val_rmse
                best_val_r2 = val_r2
            else:
                early_stop += 1
            
            logging.info('=== epoch: {}'.format(epoch))
            logging.info('Train mae: {:.5f} | Validation mae: {:.5f}, mse: {:.5f}, rmse: {:.5f}, r2: {:.5f}'.format(train_loss, val_mae, val_mse, val_rmse, val_r2))
            
            if early_stop > 50: break

        val_mae_list.append(best_val_mae)
        val_mse_list.append(best_val_mse)
        val_rmse_list.append(best_val_rmse)
        val_r2_list.append(best_val_r2)
    
    wandb.log({
        'val mae': np.mean(val_mae_list),
        'val mse': np.mean(val_mse_list),
        'val rmse': np.mean(val_rmse_list),
        'val r2': np.mean(val_r2_list),
    })
    
    # test_metrics = evaluation(model, test_loader, criterion, device, args)
    # final_test_mae = test_metrics['mae']
    # final_test_mse = test_metrics['mse']
    # final_test_rmse = test_metrics['rmse']
    # final_test_r2 = test_metrics['r2']

    logging.info('')
    logging.info('SPE')
    if args.assay_name is not None:
        logging.info(f'Assay Name: {args.assay_name}')
    elif args.tg_num is not None:
        logging.info(f'TG {args.tg_num}-{args.expose_type}')

    logging.info('Val MAE: {:.2f} ({:.2f})'.format(np.mean(val_mae_list), np.std(val_mae_list)))
    logging.info('Val MSE: {:.2f} ({:.2f})'.format(np.mean(val_mse_list), np.std(val_mse_list)))
    logging.info('Val RMSE: {:.2f} ({:.2f})'.format(np.mean(val_rmse_list), np.std(val_rmse_list)))
    logging.info('Val R2: {:.2f} ({:.2f})'.format(np.mean(val_r2_list), np.std(val_r2_list)))

    # logging.info('Test MAE: {:.2f}'.format(final_test_mae))
    # logging.info('Test MSE: {:.2f}'.format(final_test_mse))
    # logging.info('Test RMSE: {:.2f}'.format(final_test_rmse))
    # logging.info('Test R2: {:.2f}'.format(final_test_r2))

wandb.agent(sweep_id = sweep_id, function = main, count = 1000)