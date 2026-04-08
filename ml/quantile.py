import sys
sys.path.append('../')

import os
import logging
import warnings
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger

import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression

from common import (
    to_numpy_dataset,
    ParameterGrid,
    find_best_model,
    save_results
)
from modules.utils import set_seed
from modules.load_dataset import DARTDatasetREG


warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
logging.basicConfig(format='', level=logging.INFO)


def load_args():
    parser = argparse.ArgumentParser()
    
    # data arguments
    parser.add_argument('--data_path', default = '../dataset', type = str)
    parser.add_argument('--assay_name', default = None, type = str)
    parser.add_argument('--tg_num', default = None, type = int)
    parser.add_argument('--expose_type', default = None, type = str, help = 'mgkg, inhale')
    parser.add_argument('--test_size', default = 0.2, type = float)
    parser.add_argument('--random_state', default = 42, type = int)
    parser.add_argument('--fp_type', default = 'maccs', type = str, help = 'maccs, morgan, rdkit, layered, pattern')
    
    # learning arguments
    parser.add_argument('--use_smogn', default = False, type = bool)
    parser.add_argument('--use_feat_sel', default = False, type = bool)
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    return args
    

def main():
    args = load_args()
    logging.info(args)
    
    if args.tg_num is None:
        save_path = f'saved_model/{args.assay_name}'
    else: 
        save_path = f'saved_model/{args.tg_num}_{args.expose_type}'
    
    # args.assay_name = 'TOX21_SHH_3T3_GLI3_Antagonist'
    # args.fp_type = 'maccs'
    
    train_dataset = DARTDatasetREG(
        root = args.data_path, assay_name = args.assay_name, tg_num = args.tg_num, expose_type = args.expose_type,
        split = 'train', test_size = args.test_size, random_state = args.random_state
    )
    test_dataset = DARTDatasetREG(
        root = args.data_path, assay_name = args.assay_name, tg_num = args.tg_num, expose_type = args.expose_type, 
        split = 'test', test_size = args.test_size, random_state = args.random_state
    )
    
    x_tr, y_tr, x_te, y_te, _ = to_numpy_dataset(train_dataset, test_dataset, args.fp_type)
    # x_te, y_te, _ = to_numpy_dataset(test_dataset, args.fp_type)
    
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)
    y_tr = np.log10(y_tr)
    
    if args.use_feat_sel:
        mi = mutual_info_regression(x_tr, y_tr, random_state = args.random_state)
        n_keep = max(1, int(0.30 * x_tr.shape[1]))
        top_idx = np.argsort(mi)[-n_keep:]

        x_tr = x_tr[:, top_idx]
        x_te = x_te[:, top_idx]
    
    params_dict = {
        "alpha": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        "solver": ['highs'],
        'fit_intercept': [True, False]
    }
    params = ParameterGrid(params_dict)
    logging.info(f'The number of hyperparameter combinations:{len(params)}')
    
    result = {'model': {}, 'mae': {}, 'mse': {}, 'rmse': {}, 'r2': {}}
    kf = KFold(n_splits = 5, shuffle = True, random_state = args.random_state)
    
    for p in tqdm(range(len(params))):
        model_key = f'model{p}'
        result['model'][model_key] = params[p]
        result['mae'][model_key] = []
        result['mse'][model_key] = []
        result['rmse'][model_key] = []
        result['r2'][model_key] = []
        
        for train_idx, val_idx in kf.split(x_tr, y_tr):
            fold_tr_x, fold_val_x = x_tr[train_idx], x_tr[val_idx]
            fold_tr_y, fold_val_y = y_tr[train_idx], y_tr[val_idx]
            
            model = QuantileRegressor(**params[p])
            model.fit(fold_tr_x, fold_tr_y)
            pred = model.predict(fold_val_x)
            
            result['mae'][model_key].append(mean_absolute_error(fold_val_y, pred))
            result['mse'][model_key].append(mean_squared_error(fold_val_y, pred))
            result['rmse'][model_key].append(np.sqrt(mean_squared_error(fold_val_y, pred)))
            result['r2'][model_key].append(r2_score(fold_val_y, pred))
        
        save_results(result, path = save_path, file_name = f'quantile_{args.fp_type}_feat_sel_{args.use_feat_sel}.json')
            
    best_model_key, best_params, best_r2 = find_best_model(result, metric = 'r2')
    
    best_mae = np.mean(result['mae'][best_model_key])
    best_mse = np.mean(result['mse'][best_model_key])
    best_rmse = np.mean(result['rmse'][best_model_key])
    
    logging.info(f'Best Model Parameters: {best_params}')    
    logging.info(f'Validation MAE: {best_mae:.5f}')    
    logging.info(f'Validation MSE: {best_mse:.5f}')    
    logging.info(f'Validation RMSE: {best_rmse:.5f}')    
    
    final_model = QuantileRegressor(**best_params)
    final_model.fit(x_tr, y_tr)
    pred = final_model.predict(x_te)
    pred = 10 ** pred
    
    test_mae = mean_absolute_error(y_te, pred)
    test_mse = mean_squared_error(y_te, pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, pred))
    test_r2 = r2_score(y_te, pred)
    
    test_metric = {
        'mae': test_mae,
        'mse': test_mse,
        'rmse': test_rmse,
        'r2': test_r2
    }
    
    logging.info(f"Test MAE: {test_mae:.5f}")
    logging.info(f"Test MSE: {test_mse:.5f}")
    logging.info(f"Test RMSE: {test_rmse:.5f}")
    logging.info(f"Test R2: {test_r2:.5f}")
    
    checkpoints = {
        'params': final_model.get_params(),
        'metric': test_metric
    }
    
    
    file_name = f'best_quantile_{args.fp_type}_feat_sel_{args.use_feat_sel}.json'
    save_results(checkpoints, path = save_path, file_name = file_name)
    
    logging.info(f"Best model saved with R2: {test_r2:.5f}")
                

if __name__ == '__main__':
    main()
