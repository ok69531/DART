import sys
sys.path.append('../')

import logging
import warnings
from functools import partial

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from modules.utils import set_seed
from modules.load_dataset import DARTDataset

from spe_argument import load_spe_args
from spe_trainer import (
    create_mlp,
    create_mlp_ln,
    # get_snorm,
    calc_eigh,
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

dataset_path = '../dataset'
train_dataset = DARTDataset(
    dataset_path, tier = args.tier, assay_name = args.assay_name,
    split = 'train', test_size = 0.2, random_state = 42, 
    transform = partial(calc_eigh, args = args)
)
test_dataset = DARTDataset(
    dataset_path, tier = args.tier, assay_name = args.assay_name,
    split = 'test', test_size = 0.2, random_state = 42, 
    transform = partial(calc_eigh, args = args)
)


seed = 42
set_seed(seed)
num_train = int(len(train_dataset) * 0.8)
num_val = len(train_dataset) - num_train
train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train, batch_size = args.train_batch_size, shuffle = True)
val_loader = DataLoader(val, batch_size = args.val_batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = args.val_batch_size, shuffle = False)

kwargs = {}
kwargs['residual'] = args.residual
kwargs['feature_type'] = 'discrete'

criterion = nn.BCEWithLogitsLoss()
model = construct_model(args, create_mlp, **kwargs)
model.to(device)

param_groups = get_param_groups(model, args)
n_total_steps = len(train_loader) * args.n_epochs
optimizer = optim.Adam(param_groups, lr = args.lr, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, 
    lr_lambda=partial(lr_lambda, args=args, n_total_steps=n_total_steps)
)

best_val_loss, best_val_f1, best_val_auc = 100, 0, 0
final_test_loss, final_test_f1, final_test_auc = 100, 0, 0
final_test_prec, final_test_rec, final_test_auc, final_test_acc = 0, 0, 0, 0

for epoch in range(1, args.n_epochs + 1):
    train_loss = training(model, train_loader, optimizer, scheduler, criterion, device)
    val_metrics = evaluation(model, val_loader, criterion, device)
    val_loss = val_metrics['loss']; val_f1 = val_metrics['f1']; val_auc = val_metrics['auc']
    
    if val_f1 > best_val_f1:
        best_val_loss = val_loss
        best_val_f1 = val_f1
        best_val_auc = val_auc
        
        test_metrics = evaluation(model, test_loader, criterion, device)
        final_test_loss = test_metrics['loss']
        final_test_f1 = test_metrics['f1']
        final_test_prec = test_metrics['precision']
        final_test_rec = test_metrics['recall']
        final_test_auc = test_metrics['auc']
        final_test_acc = test_metrics['acc']
    
    logging.info('=== epoch: {}'.format(epoch))
    logging.info('Train loss: {:.5f} | Validation loss: {:.5f}, F1: {:.5f}, Auc: {:.5f}'.format(train_loss, val_loss, val_f1, val_auc))

logging.info('')
logging.info('SPE')
logging.info(f'Assay Name: {args.assay_name}')

logging.info('Validation Loss: {:.2f}'.format(best_val_loss))
logging.info('Validation F1-score: {:.2f}'.format(best_val_f1))
logging.info('Validation AUC: {:.2f}'.format(best_val_auc))
logging.info('Test Loss: {:.2f}'.format(final_test_loss))
logging.info('Test F1-score: {:.2f}'.format(final_test_f1))
logging.info('Test Precision: {:.2f}'.format(final_test_prec))
logging.info('Test Recall: {:.2f}'.format(final_test_rec))
logging.info('Test AUC: {:.2f}'.format(final_test_auc))
logging.info('Test ACC: {:.2f}'.format(final_test_acc))
