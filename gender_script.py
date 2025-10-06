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

now = f"{datetime.datetime.now()} BarlowTwinsLoss default"
sys.path.append("../google-research/graph_embedding/metrics")
checkpoints_path = f"gender/checkpoints_{now}"
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(f'logs/gender_{now}', exist_ok=True)


np.random.seed(42)


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

sourceA = transactions[["customer_id", "tr_datetime", "mcc_code", "term_id"]]
sourceB = transactions[["customer_id", "tr_datetime", "tr_type", "amount"]]

mcc_code_in = len(np.unique((sourceA["mcc_code"])))
term_id_in = len(np.unique((sourceA["term_id"])))
tr_type_in = len(np.unique((sourceB["tr_type"])))

print("mcc_code_in:", mcc_code_in)
print("term_id_in:", term_id_in)
print("tr_type_in", tr_type_in)

sourceA["tr_datetime"] = sourceA["tr_datetime"].apply(tr_datetime_preprocess)
sourceB["tr_datetime"] = sourceB["tr_datetime"].apply(tr_datetime_preprocess)

sourceA_preprocessor = PandasDataPreprocessor(
    col_id="customer_id",
    col_event_time="tr_datetime",
    event_time_transformation="none",
    cols_category=["mcc_code", "term_id"],
    return_records=False,
)

sourceB_preprocessor = PandasDataPreprocessor(
    col_id="customer_id",
    col_event_time="tr_datetime",
    event_time_transformation="none",
    cols_numerical=["tr_type", "amount"],
    return_records=False,
)

processed_sourceA = sourceA_preprocessor.fit_transform(sourceA)
processed_sourceB = sourceB_preprocessor.fit_transform(sourceB)

processed_sourceA.columns = [
    "sourceA_" + str(col) if str(col) != "customer_id" else str(col)
    for col in processed_sourceA.columns
]

processed_sourceB.columns = [
    "sourceB_" + str(col) if str(col) != "customer_id" else str(col)
    for col in processed_sourceB.columns
]

joined_data = processed_sourceA.merge(
    processed_sourceB,
    how="outer",
    on="customer_id"
)

joined_data = joined_data.applymap(
    lambda x: torch.tensor([]) if pd.isna(x) else x
)

train_df, test_df = train_test_split(
    joined_data,
    test_size=0.1,
    random_state=42
)
train_df, valid_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42
)

print(
    train_df.index.intersection(test_df.index)
)
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

source_features = {
    "sourceA": ["event_time", "mcc_code", "term_id"],
    "sourceB": ["event_time", "tr_type", "amount"]
}


logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(
    f"logs/gender_{now}/fraction_experiment.log")
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)

# Удалим другие обработчики
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)
logger.info("🔧 Логгер настроен вручную")

fixed_params = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "split_count": 3,
    "cnt_min": 10,
    "cnt_max": 100,
    "embedding_dim": 32,  # Размерность эмбеддингов
    "category_embedding_dim": 8,  # Размерность категорий эмбеддингов
    "hidden_size": 128,  # Размер скрытого слоя по умолчанию
    "mcc_code_in": mcc_code_in,
    "term_id_in": term_id_in,
    "tr_type_in": tr_type_in,
    "num_epochs": 30,
    'source_features': source_features,
    "loss": "ContrastiveLoss",
    "rnn_encoder_type": "gru"
}

# Список гиперпараметров для перебора
variable_params = {
    "batch_size": [16, 32, 64, 128, 256],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    "split_count": [3, 5, 7],
    "cnt_min": [5, 10, 15, 20],
    "cnt_max": [60, 80, 100, 150, 200],
    "embedding_dim": [32, 64, 128, 256, 512, 1024],
    "category_embedding_dim": [4, 8, 16, 24, 32, 64, 128],
    "hidden_size": [64, 128, 256, 512, 1024, 2048, 4096],
    "loss": ["BarlowTwinsLoss", "ContrastiveLoss", "VicregLoss", "SoftmaxLoss"],
    "rnn_encoder_type": ["gru", "lstm"]
}

all_hyperparameter_grids = create_params_grid(fixed_params, variable_params)


out_folder = f"/home/dpetrovitch/dzagcoffee/output_{now}"
out_prefix = out_folder + f"/out"

os.makedirs(out_folder, exist_ok=True)

sample_fractions = np.linspace(1/20, 1, 5)

run_grid_search(
    all_hyperparameter_grids,
    sample_fractions,
    train_dict,
    valid_dict,
    test_dict,
    targets,
    checkpoints_path,
    logger,
    col_id="customer_id",
    target_col='gender',
    out_prefix=out_prefix,
    verbose=0,
    n_samples=1,
    downstream_type="logreg"  # 'catboost' | 'mlp' | 'logreg'
)
