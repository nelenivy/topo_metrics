from run_exp import create_params_grid, run_grid_search
import sys
import os
import logging
import pandas as pd
import numpy as np
import datetime
from ptls.preprocessing import PandasDataPreprocessor
import torch
from sklearn.model_selection import train_test_split


# "BarlowTwinsLoss" | "ContrastiveLoss" | "VicregLoss" | "SoftmaxLoss"
DEFAULT_LOSS = "ContrastiveLoss"

# 'catboost' | 'mlp' | 'logreg'
DEFAULT_DOWNSTEREAM = "catboost"

# "lstm" | "gru"
DEFAULT_BACKBONE = "lstm"

now = f"{datetime.datetime.now()}"
sys.path.append("/home/dpetrovitch/dzagcoffee/topo_metrics/google-research")

checkpoints_path = f"../train_trace/gender/checkpoints_{now}"
os.makedirs(checkpoints_path, exist_ok=True, )
os.makedirs(
    f'../train_trace/logs/gender_{now}_{DEFAULT_LOSS}_{DEFAULT_BACKBONE}_{DEFAULT_DOWNSTEREAM}',
    exist_ok=True
)

np.random.seed(42)


logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(
    f"../train_trace/logs/gender_{now}_{DEFAULT_LOSS}_{DEFAULT_BACKBONE}_{DEFAULT_DOWNSTEREAM}/fraction_experiment.log"
)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

file_handler.setFormatter(formatter)

if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)
logger.info("🔧 Логгер настроен вручную")


def tr_datetime_preprocess(tr_datetime):
    days, hms = tr_datetime.split()
    hh, mm, ss = hms.split(":")

    seconds = datetime.timedelta(
        hours=int(hh), minutes=int(mm), seconds=int(ss))
    seconds = seconds.total_seconds()
    seconds += int(days) * 24 * 3600

    return int(seconds)


transactions = pd.read_csv(
    "https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/transactions.csv.gz?download=true",
    compression="gzip"
)
targets = pd.read_csv(
    "https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/gender_train.csv?download=true"
)

transactions = transactions.dropna().reset_index(drop=True)

n_cutomers = len(pd.unique(transactions["customer_id"]))
n_labeling_cutomers = len(pd.unique(targets["customer_id"]))

mcc_code_in = len(np.unique((transactions["mcc_code"])))
term_id_in = len(np.unique((transactions["term_id"])))
tr_type_in = len(np.unique((transactions["tr_type"])))

print("mcc_code_in:", mcc_code_in)
print("term_id_in:", term_id_in)
print("tr_type_in", tr_type_in)

transactions["tr_datetime"] = transactions["tr_datetime"].apply(
    tr_datetime_preprocess
)

preprocessor = PandasDataPreprocessor(
    col_id="customer_id",
    col_event_time="tr_datetime",
    event_time_transformation="none",
    cols_category=["tr_type", "mcc_code", "term_id"],
    cols_numerical=["amount"],
    return_records=False,
)

transactions = preprocessor.fit_transform(transactions)

transactions = transactions.map(
    lambda x: torch.tensor([]) if pd.isna(x) else x
)

train_df, test_df = train_test_split(
    transactions,
    test_size=0.1,
    random_state=42
)
train_df, valid_df = train_test_split(
    transactions,
    test_size=0.1,
    random_state=42
)

print(train_df.index.intersection(test_df.index))
print(
    train_df['customer_id'].unique().shape,
    test_df['customer_id'].unique().shape
)
print(
    np.unique(test_df.index.values).shape,
    test_df.shape
)
print(test_df.index)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_dict = train_df.to_dict("records")
valid_dict = valid_df.to_dict("records")
test_dict = test_df.to_dict("records")


fixed_params = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "split_count": 3,
    "cnt_min": 10,
    "cnt_max": 100,
    "embedding_dim": 32,
    "category_embedding_dim": 8,
    "hidden_size": 128,
    "mcc_code_in": mcc_code_in,
    "term_id_in": term_id_in,
    "tr_type_in": tr_type_in,
    "num_epochs": 1,
    "loss": DEFAULT_LOSS,
    "rnn_encoder_type": DEFAULT_BACKBONE
}

variable_params = {
    "batch_size": [16]  # , 32, 64, 128, 256],
    # "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    # "split_count": [3, 5, 7],
    # "cnt_min": [5, 10, 15, 20],
    # "cnt_max": [60, 80, 100, 150, 200],
    # "embedding_dim": [32, 64, 128, 256, 512, 1024],
    # "category_embedding_dim": [4, 8, 16, 24, 32, 64, 128],
    # "hidden_size": [64, 128, 256, 512, 1024, 2048, 4096],
    # "loss": ["BarlowTwinsLoss", "ContrastiveLoss", "VicregLoss", "SoftmaxLoss"],
    # "rnn_encoder_type": ["gru", "lstm"]
}

all_hyperparameter_grids = create_params_grid(fixed_params, variable_params)


out_folder = f"/home/dpetrovitch/dzagcoffee/output_{now}_{DEFAULT_LOSS}_{DEFAULT_BACKBONE}_{DEFAULT_DOWNSTEREAM}"
out_prefix = out_folder + f"/out"

os.makedirs(out_folder, exist_ok=True)

sample_fractions = np.linspace(1/20, 1, 5)

run_grid_search(
    all_hyperparameter_grids=all_hyperparameter_grids,
    sample_fractions=sample_fractions,
    train_data_in=train_dict,
    valid_data_in=valid_dict,
    test_data_in=test_dict,
    targets=targets,
    checkpoints_path=checkpoints_path,
    logger=logger,
    col_id="customer_id",
    target_col='gender',
    out_prefix=out_prefix,
    verbose=0,
    n_samples=1,
    downstream_type=DEFAULT_DOWNSTEREAM,
    devices=[1]
)
