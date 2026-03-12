import os
import json
import torch
import numpy as np
import pandas as pd
from itertools import product
from collections.abc import Iterable


def _extract_parts(dataset, fing_type):
    y_list, fing_list, desc_list = [], [], []

    for data in dataset:
        # y
        y_list.append(data.y)

        # fingerprint
        fp = data[fing_type]
        if isinstance(fp, torch.Tensor):
            fp = fp.cpu().numpy()
        else:
            fp = np.asarray(fp)
        fing_list.append(fp)

        # descriptor
        desc_row = {}
        for k, v in data.descriptor.items():
            if isinstance(v, torch.Tensor):
                desc_row[k] = v.item()
            else:
                desc_row[k] = v
        desc_list.append(desc_row)

    y = torch.cat(y_list).cpu().numpy()
    fing = np.stack(fing_list)
    desc_df = pd.DataFrame(desc_list)

    return y, fing, desc_df


def to_numpy_dataset(train_dataset, test_dataset, fing_type):
    y_tr, fing_tr, desc_tr = _extract_parts(train_dataset, fing_type)
    y_te, fing_te, desc_te = _extract_parts(test_dataset, fing_type)

    # train/test를 합쳐서 NaN 있는 컬럼을 공통으로 제거
    desc_all = pd.concat([desc_tr, desc_te], axis=0, ignore_index=True)
    valid_cols = desc_all.columns[~desc_all.isna().any(axis=0)].tolist()

    desc_tr = desc_tr[valid_cols].to_numpy(dtype=float)
    desc_te = desc_te[valid_cols].to_numpy(dtype=float)

    x_tr = np.concatenate([fing_tr, desc_tr], axis=1)
    x_te = np.concatenate([fing_te, desc_te], axis=1)

    return x_tr, y_tr, x_te, y_te, valid_cols


# def to_numpy_dataset(dataset, fing_type):
#     y, fing, desc = ([] for _ in range(3))

#     for data in dataset:
#         y.append(data.y)
#         fing.append(data[fing_type])
#         desc.append({k: v.item() for k, v in data.descriptor.items()})
#     y = torch.cat(y).numpy()
#     fing = np.stack(fing)
#     desc = pd.DataFrame(desc)
#     descriptor_names = list(desc.columns)
#     desc = desc.dropna(axis = 1).to_numpy()
#     x = np.concatenate([fing, desc], axis = 1)
    
#     return x, y, descriptor_names


def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v)))

    return params_grid


def find_best_model(results, metric='rmse', metric_agg='mean'):
    best_model = None
    best_score = -np.inf
    best_model_key = None
    
    for model_key in results['model'].keys():
        scores = results[metric][model_key]
        if metric_agg == 'mean':
            agg_score = np.mean(scores)
        elif metric_agg == 'median':
            agg_score = np.median(scores)
        else:
            raise ValueError("metric_agg must be either 'mean' or 'median'")
        
        if agg_score > best_score:
            best_score = agg_score
            best_model = results['model'][model_key]
            best_model_key = model_key
    
    return best_model_key, best_model, best_score


def save_results(result, path, file_name):
    if not os.path.exists(path): os.makedirs(path)
    with open(os.path.join(path, file_name), 'w') as f:
        json.dump(result, f)
    