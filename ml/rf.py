import sys
sys.path.append('../')

import os
import logging
import warnings
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    # root_mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor
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
logging.basicConfig(format='', level=logging.INFO)


def load_args():
    parser = argparse.ArgumentParser()
    
    # data arguments
    parser.add_argument('--data_path', default = '../dataset', type = str)
    parser.add_argument('--assay_aname', default = None)
    parser.add_argument('--tg_num', default = None)
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

    train_dataset = DARTDatasetREG(
        root = args.data_path, assay_name = args.assay_name, tg_num = args.tg_num, 
        split = 'train', test_size = args.test_size, random_state = args.random_state
    )
    test_dataset = DARTDatasetREG(
        root = args.data_path, assay_name = args.assay_name, tg_num = args.tg_num, 
        split = 'test', test_size = args.test_size, random_state = args.random_state
    )
    
    x_tr, y_tr, _ = to_numpy_dataset(train_dataset, args.fp_type)
    x_te, y_te, _ = to_numpy_dataset(test_dataset, args.fp_type)
    
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
        "n_estimators": [5, 10, 30, 50, 100, 300, 500, 800],
        "criterion": ['absolute_error', 'friedman_mse'],
        "max_depth": [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [1.0, 0.5, "sqrt"],
        "max_samples": [None, 0.7, 0.85],
    }
    params = ParameterGrid(params_dict)
    
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
            
            model = RandomForestRegressor(random_state = args.random_state, **params[p])
            model.fit(fold_tr_x, fold_tr_y)
            pred = model.predict(fold_val_x)
            
            # print('scaled mse', mean_squared_error(fold_val_y, pred))
            # print('scaled rmse', np.sqrt(mean_squared_error(fold_val_y, pred)))
            # print('scaled mae', mean_absolute_error(fold_val_y, pred))
            # print('scaled mae', r2_score(fold_val_y, pred))
            
            result['mae'][model_key].append(mean_absolute_error(fold_val_y, pred))
            result['mse'][model_key].append(mean_squared_error(fold_val_y, pred))
            result['rmse'][model_key].append(np.sqrt(mean_squared_error(fold_val_y, pred)))
            result['r2'][model_key].append(r2_score(fold_val_y, pred))
        
        save_results(result, path = 'saved_model', file_name = f'rf_{args.fp_type}_feat_sel_{args.use_feat_sel}.json')
        
    best_model_key, best_params, best_r2 = find_best_model(result, metric = 'r2')
    
    best_mae = np.mean(result['mae'][best_model_key])
    best_mse = np.mean(result['mse'][best_model_key])
    best_rmse = np.mean(result['rmse'][best_model_key])
    
    logging.info(f'Best Model Parameters: {best_params}')    
    logging.info(f'Validation MAE: {best_mae:.5f}')    
    logging.info(f'Validation MSE: {best_mse:.5f}')    
    logging.info(f'Validation RMSE: {best_rmse:.5f}')    
    
    final_model = RandomForestRegressor(random_state = args.random_state, **best_params)
    final_model.fit(x_tr, y_tr)
    pred = final_model.predict(x_te)
    pred = 10 ** pred
    
    test_mae = mean_absolute_error(y_te, pred)
    test_mse = mean_squared_error(y_te, pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, pred))
    test_r2 = r2_score(y_te, pred)
    
    logging.info(f"Test MAE: {test_mae:.5f}")
    logging.info(f"Test MSE: {test_mse:.5f}")
    logging.info(f"Test RMSE: {test_rmse:.5f}")
    logging.info(f"Test R2: {test_r2:.5f}")
    
    save_results(final_model.get_params(), path = 'saved_model', file_name = f'best_rf_{args.fp_type}_feat_sel_{args.use_feat_sel}.json')
    logging.info(f"Best model saved with R2: {test_r2:.5f}")
                


# pairplot
# import seaborn as sns
    
# for i in range(40):
#     arr = np.concatenate(
#         [y.reshape(-1, 1), x[:, 167 + (i*5):167 + ((i+1)*5)]],
#         axis=1
#     )

#     df_plot = pd.DataFrame(arr)

#     # 전부 숫자로 강제 변환
#     df_plot = df_plot.apply(pd.to_numeric, errors='coerce')

#     # inf, -inf 제거
#     df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()
    
#     sns.pairplot(
#         df_plot,
#         diag_kind='hist',
#         diag_kws={'bins': 30},
#         corner=True
#     )
