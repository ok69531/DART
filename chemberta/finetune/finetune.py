"""Script for finetuning and evaluating pre-trained ChemBERTa models on MoleculeNet tasks.

[classification]
python finetune.py --datasets=bbbp --pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015

[regression]
python finetune.py --datasets=delaney --pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015

[csv]
python finetune.py --datasets=$HOME/finetune_datasets/logd/ \
                --dataset_types=regression \
                --pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015 \
                --is_molnet=False

[multiple]
python finetune.py \
--datasets=bace_classification,bace_regression,bbbp,clearance,clintox,delaney,lipo,tox21 \
--pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015 \
--n_trials=20 \
--output_dir=finetuning_experiments \
--run_name=sm_015

[from scratch (no pretraining)]
python finetune.py --datasets=bbbp

"""
import sys
sys.path.append('../')
sys.path.append('../../')

import json
import os
import shutil
import argparse

from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from absl import app, flags
from torch.utils.data import random_split
from chemberta.utils.molnet_dataloader import get_dataset_info, load_molnet_dataset
from chemberta.utils.roberta_regression import (
    RobertaForRegression,
    RobertaForSequenceClassification,
)
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from transformers import RobertaConfig, RobertaTokenizerFast, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback


def load_argument():
    parser = argparse.ArgumentParser(description="Training script configuration")

    # Settings
    parser.add_argument("--output_dir", type=str, default='../../saved_result/chemberta', help="")
    parser.add_argument("--overwrite_output_dir", type = bool, default = True, help="If set, overwrite existing output directory.")
    parser.add_argument("--run_name", type=str, default="base", help="")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")

    # Model params
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="../pretrained_cks/base",
        help="Path to pretrained model or HuggingFace model ID."
    )
    parser.add_argument(
        "--freeze_base_model", type=bool, default=False, help="If set, freezes base model parameters during training."
    )
    parser.add_argument(
        "--is_molnet",
        action="store_true",
        help="If set, assumes all datasets are MolNet datasets."
    )

    # RobertaConfig params
    parser.add_argument("--vocab_size", type=int, default=600, help="")
    parser.add_argument("--max_position_embeddings", type=int, default=515, help="")
    parser.add_argument("--num_attention_heads", type=int, default=6, help="")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="")
    parser.add_argument("--type_vocab_size", type=int, default=1, help="")

    # Train params
    parser.add_argument("--logging_steps", type=int, default=10, help="")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="")
    parser.add_argument("--num_train_epochs_max", type=int, default=10, help="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64, help="")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64, help="")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of hyperparameter combinations to try."
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds for best model evaluation."
    )

    # Dataset params
    parser.add_argument(
        "--datasets",
        type=str,
        default='dart_tg414',
        help="Comma-separated list of dataset names."
    )
    parser.add_argument("--split", type=str, default="scaffold", help="Data split type.")
    parser.add_argument(
        "--dataset_types",
        type=str,
        default="classification",
        help="Comma-separated list of dataset types."
    )

    # Tokenizer params
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="../pretrained_cks/base",
        help="Path to tokenizer."
    )
    parser.add_argument("--max_tokenizer_len", type=int, default=512, help="")

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    # 문자열 리스트 처리 (datasets, dataset_types)
    # args.datasets = [d.strip() for d in args.datasets.split(",")]
    # args.dataset_types = [d.strip() for d in args.dataset_types.split(",")]

    return args

args = load_argument()


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_DISABLED"] = "true"


def main():
    if args.pretrained_model_name_or_path is None:
        print(
            "`WARNING: pretrained_model_name_or_path` is None - training a model from scratch."
        )
    else:
        print(
            f"Instantiating pretrained model from: {args.pretrained_model_name_or_path}"
        )

    is_molnet = args.is_molnet

    # for i in range(len(FLAGS.datasets)):
    # dataset_name_or_path = args.datasets[i]
    # dataset_name = get_dataset_name(dataset_name_or_path)
    dataset_name = args.datasets
    dataset_type = (
        get_dataset_info(dataset_name)["dataset_type"]
        if is_molnet
        else args.dataset_types
    )

    run_dir = os.path.join(args.output_dir, args.run_name + '_' + dataset_name.split('_')[-1])
    

    if os.path.exists(run_dir) and not args.overwrite_output_dir:
        print(f"Run dir already exists for dataset: {dataset_name}")
    else:
        print(f"Finetuning on {dataset_name}")
        finetune_single_dataset(
            dataset_name, dataset_type, run_dir, is_molnet
        )


def prune_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        return None

    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    try:
        loaded_state_dict = torch.load(state_dict_path)
    except:
        loaded_state_dict = torch.load(state_dict_path, map_location='cpu')
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict


def finetune_single_dataset(dataset_name, dataset_type, run_dir, is_molnet):
    torch.manual_seed(args.seed)

    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.tokenizer_path, max_len=args.max_tokenizer_len, use_auth_token=True
    )

    finetune_datasets = get_finetune_datasets(dataset_name, tokenizer, is_molnet)

    if args.pretrained_model_name_or_path:
        config = RobertaConfig.from_pretrained(
            args.pretrained_model_name_or_path, use_auth_token=True
        )
    else:
        config = RobertaConfig(
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            type_vocab_size=args.type_vocab_size,
            is_gpu=torch.cuda.is_available(),
        )

    if dataset_type == "classification":
        model_class = RobertaForSequenceClassification
        config.num_labels = finetune_datasets.num_labels

    elif dataset_type == "regression":
        model_class = RobertaForRegression
        config.num_labels = 1
        config.norm_mean = finetune_datasets.norm_mean
        config.norm_std = finetune_datasets.norm_std

    state_dict = prune_state_dict(args.pretrained_model_name_or_path)

    def model_init():
        if dataset_type == "classification":
            model_class = RobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = RobertaForRegression

        if args.pretrained_model_name_or_path:
            model = model_class.from_pretrained(
                args.pretrained_model_name_or_path,
                config=config,
                state_dict=state_dict,
                use_auth_token=True,
            )
            if args.freeze_base_model:
                for name, param in model.base_model.named_parameters():
                    param.requires_grad = False
        else:
            model = model_class(config=config)

        return model

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir=run_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        report_to=None,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=finetune_datasets.train_dataset,
        eval_dataset=finetune_datasets.valid_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    def custom_hp_space_optuna(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs", 1, args.num_train_epochs_max
            ),
            "seed": trial.suggest_int("seed", 1, 10),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [args.per_device_train_batch_size]
            ),
        }

    best_trial = trainer.hyperparameter_search(
        backend="optuna",
        direction="minimize",
        hp_space=custom_hp_space_optuna,
        n_trials=args.n_trials,
    )

    # Set parameters to the best ones from the hp search
    for n, v in best_trial.hyperparameters.items():
        setattr(trainer.args, n, v)

    dir_valid = os.path.join(run_dir, "valid")
    dir_test = os.path.join(run_dir, "test")
    os.makedirs(dir_valid, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    metrics_valid = {}
    metrics_test = {}

    # Run with several seeds so we can see std
    for random_seed in range(args.n_seeds):
        setattr(trainer.args, "seed", random_seed)
        trainer.train()
        metrics_valid[f"seed_{random_seed}"] = eval_model(
            trainer,
            finetune_datasets.valid_dataset_unlabeled,
            dataset_name,
            dataset_type,
            dir_valid,
            random_seed,
        )
        metrics_test[f"seed_{random_seed}"] = eval_model(
            trainer,
            finetune_datasets.test_dataset,
            dataset_name,
            dataset_type,
            dir_test,
            random_seed,
        )

    with open(os.path.join(dir_valid, "metrics.json"), "w") as f:
        json.dump(metrics_valid, f)
    with open(os.path.join(dir_test, "metrics.json"), "w") as f:
        json.dump(metrics_test, f)

    # Delete checkpoints from hyperparameter search since they use a lot of disk
    for d in glob(os.path.join(run_dir, "run-*")):
        shutil.rmtree(d, ignore_errors=True)


def eval_model(trainer, dataset, dataset_name, dataset_type, output_dir, random_seed):
    labels = dataset.labels
    predictions = trainer.predict(dataset)
    # fig = plt.figure(dpi=144)

    if dataset_type == "classification":
        if len(np.unique(labels)) <= 2:
            y_pred_score = softmax(predictions.predictions, axis=1)[:, 1]
            y_pred = np.argmax(predictions.predictions, axis = 1)
            metrics = {
                "roc_auc_score": roc_auc_score(y_true=labels, y_score=y_pred),
                'precision_score': precision_score(y_true=labels, y_pred=y_pred),
                'recall_score': recall_score(y_true=labels, y_pred=y_pred),
                'f1_score': f1_score(y_true=labels, y_pred=y_pred),
                'accuracy_score': accuracy_score(y_true=labels, y_pred=y_pred),
                "average_precision_score": average_precision_score(
                    y_true=labels, y_score=y_pred
                ),
            }
            # sns.histplot(x=y_pred, hue=labels)
        else:
            y_pred = np.argmax(predictions.predictions, axis=-1)
            metrics = {"mcc": matthews_corrcoef(labels, y_pred)}

    elif dataset_type == "regression":
        y_pred = predictions.predictions.flatten()
        metrics = {
            "pearsonr": pearsonr(y_pred, labels),
            "rmse": mean_squared_error(y_true=labels, y_pred=y_pred, squared=False),
        }
        # sns.regplot(x=y_pred, y=labels)
        # plt.xlabel("ChemBERTa predictions")
        # plt.ylabel("Ground truth")
    else:
        raise ValueError(dataset_type)

    # plt.title(f"{dataset_name} {dataset_type} results")
    # plt.savefig(os.path.join(output_dir, f"results_seed_{random_seed}.png"))

    return metrics


def get_finetune_datasets(dataset_name, tokenizer, is_molnet=False):
    # if is_molnet:
    #     tasks, (train_df, valid_df, test_df), _ = load_molnet_dataset(
    #         dataset_name, split=FLAGS.split, df_format="chemprop"
    #     )
    #     assert len(tasks) == 1
    # else:
    #     train_df = pd.read_csv(os.path.join(dataset_name, "train.csv"))
    #     valid_df = pd.read_csv(os.path.join(dataset_name, "valid.csv"))
    #     test_df = pd.read_csv(os.path.join(dataset_name, "test.csv"))
    
    df = pd.read_excel(f'../../dataset/{dataset_name}.xlsx')
    num_train = int(len(df) * 0.8)
    num_valid = int(len(df) * 0.1)
    num_test = len(df) - (num_train + num_valid)
    assert num_train + num_valid + num_test == len(df)

    indices = torch.arange(len(df))
    train_idx, valid_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])
    
    train_df = df.iloc[train_idx.indices]
    valid_df = df.iloc[valid_idx.indices]
    test_df = df.iloc[test_idx.indices]

    train_dataset = FinetuneDataset(train_df, tokenizer)
    valid_dataset = FinetuneDataset(valid_df, tokenizer)
    valid_dataset_unlabeled = FinetuneDataset(valid_df, tokenizer, include_labels=False)
    test_dataset = FinetuneDataset(test_df, tokenizer, include_labels=False)

    num_labels = len(np.unique(train_dataset.labels))
    norm_mean = [np.mean(np.array(train_dataset.labels), axis=0)]
    norm_std = [np.std(np.array(train_dataset.labels), axis=0)]

    return FinetuneDatasets(
        train_dataset,
        valid_dataset,
        valid_dataset_unlabeled,
        test_dataset,
        num_labels,
        norm_mean,
        norm_std,
    )


# def get_dataset_name(dataset_name_or_path):
#     return os.path.splitext(os.path.basename(dataset_name_or_path))[0]


@dataclass
class FinetuneDatasets:
    train_dataset: str
    valid_dataset: torch.utils.data.Dataset
    valid_dataset_unlabeled: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    num_labels: int
    norm_mean: List[float]
    norm_std: List[float]


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, include_labels=True):

        self.encodings = tokenizer(df["SMILES"].tolist(), truncation=True, padding=True)
        self.labels = df.toxicity.values
        self.include_labels = include_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.include_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


if __name__ == "__main__":
    main()
