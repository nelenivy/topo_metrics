import os
import gc
import torch
import shutil
import pandas as pd

from time import time
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from run_models import ModelKeeper
from run_metrics import evaluate_one_emb


def clear_checkpoints_dir(checkpoints_path: str) -> None:
    if not os.path.exists(checkpoints_path):
        return

    for name in os.listdir(checkpoints_path):
        path = os.path.join(checkpoints_path, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Не удалось удалить {path}: {e}")


def create_truncated_params_grid(
    fixed_params: Dict[str, Any],
    variable_params: Dict[str, List[Any]]
) -> List[Tuple[str, Dict[str, Any]]]:
    grids = []
    for param_name, values in variable_params.items():
        for value in values:
            grid = {**fixed_params, param_name: value}
            grids.append((param_name, grid))
    return grids


def create_full_params_grid(
    fixed_params: Dict[str, Any],
    variable_params: Dict[str, List[Any]]
) -> List[Tuple[str, Dict[str, Any]]]:
    if not variable_params:
        return [("none", fixed_params.copy())]

    param_names = list(variable_params.keys())
    all_combinations = product(
        *(variable_params[name] for name in param_names)
    )

    grids = []
    for combination in all_combinations:
        grid = {**fixed_params, **dict(zip(param_names, combination))}
        grids.append((", ".join(param_names), grid))

    return grids


def run_grid_search(
    all_hyperparameter_grids: List[Tuple[str, Dict[str, Any]]],
    sample_fractions: Tuple[float, ...],
    train_data_in: pd.DataFrame,
    valid_data_in: pd.DataFrame,
    test_data_in: pd.DataFrame,
    targets: pd.DataFrame,
    checkpoints_path: str,
    logger,
    col_id: str = "customer_id",
    target_col: str = "gender",
    out_prefix: str | None = None,
    verbose: int = 0,
    n_samples: int = 10,
    downstream_type: str = "catboost",
    devices: int | str = 0
) -> None:
    start_time = time()
    all_embeddings = []

    for param_name, params in tqdm(all_hyperparameter_grids, desc="Grid search"):
        logger.info(f"Тестируем: изменяется только {param_name}")
        logger.info(f"Параметры: {params}")

        torch.cuda.empty_cache()
        gc.collect()

        model_keeper = ModelKeeper()
        model_keeper.create_datasets(
            train_data_in=train_data_in,
            valid_data_in=valid_data_in,
            params=params,
            col_id=col_id
        )
        model_keeper.train_model(
            params=params,
            checkpoints_path=checkpoints_path,
            devices=devices
        )

        embs = model_keeper.calc_embs_from_trained(test_data_in)
        all_embeddings.extend(embs)

        clear_checkpoints_dir(checkpoints_path)

    logger.info(
        f"Grid search завершён за {round(time() - start_time, 2)} сек."
    )

    eval_many_embs(
        embs_list=all_embeddings,
        targets=targets,
        col_id=col_id,
        target_col=target_col,
        out_prefix=out_prefix,
        sample_fractions=sample_fractions,
        verbose=verbose,
        n_samples=n_samples,
        downstream_type=downstream_type
    )


def eval_many_embs(
    embs_list: List[Dict[str, Any]],
    targets: pd.DataFrame,
    col_id: str = "customer_id",
    target_col: str = "gender",
    out_prefix: str | None = None,
    sample_fractions: Tuple[float, ...] = (1 / 20,),
    verbose: int = 0,
    n_samples: int = 10,
    downstream_type: str = "catboost"
) -> None:

    results = defaultdict(list)

    for curr_emb in tqdm(embs_list, desc="Evaluating embeddings"):
        res = evaluate_one_emb(
            curr_emb["emb"],
            targets,
            sample_fractions=sample_fractions,
            col_id=col_id,
            target_col=target_col,
            verbose=verbose,
            n_samples=n_samples,
            downstream_type=downstream_type
        )

        for metrics in res:
            sample_frac = metrics["sample_fraction"]
            metrics_flat = {
                k: round(v, 4) for k, v in metrics.items() if k != "sample_fraction"}
            result_row = {**curr_emb["info"], **metrics_flat}
            results[sample_frac].append(result_row)

    for sample_frac, result_list in results.items():
        df = pd.DataFrame(result_list)
        output_csv = f"{out_prefix}_{sample_frac:.3f}".rstrip(
            "0").rstrip(".") + ".csv"
        df.to_csv(output_csv, index=False)

        print(
            f"Сохранён результат для выборки {sample_frac:.3f}: {output_csv}")
