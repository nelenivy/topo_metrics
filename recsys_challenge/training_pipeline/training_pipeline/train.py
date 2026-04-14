import argparse
import logging

from pytorch_lightning.callbacks import EarlyStopping

from itertools import chain

from pathlib import Path

from training_pipeline.tasks import (
    ChurnTasks,
    PropensityTasks,
    parse_task,
)
from training_pipeline.task_constructor import (
    TaskConstructor,
)
from training_pipeline.logger_factory import (
    NeptuneLoggerFactory,
)
from training_pipeline.train_runner import (
    run_tasks,
)
from data_utils.data_dir import DataDir
from typing import List

from clearml import Task

import sys
import pandas as pd
import numpy as np
import ripserplusplus as rpp
from scipy.spatial.distance import pdist, squareform
import math
from tqdm import tqdm
from sklearn.utils import resample
from time import time
from .topology import calculate_ph_dim
from .unsupervised import (rankme,
        coherence,
        pseudo_condition_number,
        alpha_req,
        stable_rank,
        ne_sum,
        self_clustering)
import gc


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def ripser_metric(embeddings, u=None, s=None):    
    diagrams = rpp.run("--format point-cloud", embeddings)
    persistence = {}
    distances = pdist(embeddings)
    distance_matrix = squareform(distances)
    sorted_rows = np.sort(distance_matrix, axis=1)
    mean_nearest_dist = sorted_rows[:, 10].mean()
    mean_largest_dist = sorted_rows[:, -10].mean()
    distances_arr = distance_matrix.ravel()
    quants = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    norms = list(np.quantile(distances_arr, quants)) + [mean_nearest_dist, mean_largest_dist]
    quants += ['mean_10', "mean_last_10"]
    
    for k in range(len(diagrams)):
        pers_lens = [death - birth for birth, death in diagrams[k] if death > birth]
        persistence_sum = sum(pers_lens)
        persistence[f"ripser_sum_H{k}"] = persistence_sum
        persistence_sq_sum = sum([l ** 2 for l in pers_lens])
        persistence[f"ripser_log_sum{k}"] = sum([np.log(1.0 + l) for l in pers_lens])
        persistence[f"ripser_norm_sum{k}"] = sum([(death - birth) / (death + birth)
                                    for birth, death in diagrams[k] if death > birth])
        persistence[f"ripser_log_sum_norm{k}"] = sum([np.log(1.0 + (death - birth) / (death + birth))
                                    for birth, death in diagrams[k] if death > birth])
        
        persistence[f"ripser_sq_sum_H{k}"] = math.sqrt(persistence_sq_sum)
        
        for q, v in zip(quants, norms):
            persistence[f"ripser_sum_H{k}_norm{q}"] = persistence[f"ripser_sum_H{k}"] / v
            persistence[f"ripser_sq_sum_H{k}_norm{q}"] = persistence[f"ripser_sq_sum_H{k}"] / v
            persistence[f"ripser_log_sum{k}_norm{q}"] = persistence[f"ripser_log_sum{k}"] / np.log(1.0 + v)
        #persistence["ripser_sum"]+= persistence_sum

    return persistence


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory where target and input data are stored",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory where input embeddings are stored",
    )
    valid_tasks = " ".join([task.value for task in chain(ChurnTasks, PropensityTasks)])
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        required=True,
        help=f"Name of the task to train out of: {valid_tasks}",
    )
    parser.add_argument("--log-name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading",
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        required=True,
        type=str,
        help='List of devices to use. Possible options: "auto", id of single device to use or list of ids of devices to use.',
    )
    parser.add_argument(
        "--neptune-api-token",
        required=False,
        type=str,
        help="Neptune API token.",
    )

    parser.add_argument(
        "--neptune-project",
        required=False,
        type=str,
        help="Name of Neptune project within workspace to save result to.",
    )
    parser.add_argument(
        "--score-dir",
        required=False,
        type=str,
        help="Path to directory where to save best scores for each task",
    )
    parser.add_argument(
        "--disable-relevant-clients-check",
        action="store_true",
        help="Disables relevant clients check in validator, but enables embeddings for sets of clients other than relevant clients.",
    )

    parser.add_argument(
        "--clearml-project",
        required=False,
        type=str,
        help="Name of ClearML project to save result to.",
    )

    return parser


def compute_metrics(embeddings_np, selected_metrics=None, 
        n_samples=5, sample_fraction=1/20, clearml_task=None):    
    sample_size = max(1, int(sample_fraction * embeddings_np.shape[0]))

    available_metrics = {
        "rankme": rankme,
        "coherence": coherence,
        "pseudo_condition_number": pseudo_condition_number,
        "alpha_req": alpha_req,
        "stable_rank": stable_rank,
        "ne_sum": ne_sum,
        "self_clustering": self_clustering,
        "ripser": ripser_metric,
        "ph_dim": calculate_ph_dim
    }
    if selected_metrics is None:
        selected_metrics = list(available_metrics.keys())

    metrics = {name: [] for name in selected_metrics}
    times = {name: [] for name in selected_metrics}

    for i in tqdm(range(n_samples)):
        sample = resample(embeddings_np, n_samples=sample_size, replace=False, random_state=42 + i)
        u, s, _ = np.linalg.svd(sample, compute_uv=True, full_matrices=False)

        for metric_name in tqdm(selected_metrics):
            if metric_name not in available_metrics:
                continue

            try:
                t0 = time()
                result = available_metrics[metric_name](sample, u=u, s=s)
                t = time() - t0

                if isinstance(result, dict):
                    for subname, val in result.items():
                        if subname not in metrics:
                            metrics[subname] = []
                            times[subname] = []
                        metrics[subname].append(val)
                        times[subname].append(t)
                else:
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                        times[metric_name] = []
                    metrics[metric_name].append(result)
                    times[metric_name].append(t)
            except Exception as e:
                print(f"⚠️ Failed to compute {metric_name} on sample {i}: {e}")

        gc.collect()

    averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}

    if clearml_task:
        clearml_logger = clearml_task.get_logger()
        for key, value in averaged_metrics.items():
            clearml_logger.report_single_value(f'mean_{key}', value)
    
        for key, value in std_metrics.items():
            clearml_logger.report_single_value(f'std_{key}', value)
    
    averaged_times = {k: np.mean(v) for k, v in times.items()}
    std_times = {k: np.std(v) for k, v in times.items()}

    if clearml_task:
        for key, value in averaged_times.items():
            clearml_logger.report_single_value(f'mean_time_{key}', value)
    
        for key, value in std_times.items():
            clearml_logger.report_single_value(f'std_time_{key}', value)

    print("\n📊 Mean:")
    for metric_name in averaged_metrics:
        metric_value = averaged_metrics[metric_name]
        metric_time = averaged_times.get(metric_name, None)
        print(f"🧠 {metric_name:30s} = {metric_value:.4f} | ⏱ {metric_time:.4f} сек")

    return averaged_metrics, averaged_times


def parse_devices(device_arg: List[str]) -> List[int] | int | str:
    """
    Method to parse --devices argument of argparse and return devices to use.
    Args:

        device_arg (List[str]): --devices command line argument from argparse

    Returns:
        List[int] | int | str : devices to use: "auto", a single device id or list of device ids.
    """
    if (len(device_arg) == 1) and (device_arg[0] == "auto"):
        return "auto"
    else:
        try:
            return [int(device) for device in device_arg]
        except ValueError:
            raise ValueError(
                f'Devices argument should be one one of "auto", int or list of ints, received: "{" ".join(device_arg)}"'
            )


def main(params) -> None:

    clearml_task = Task.init(project_name=params.clearml_project,
                             task_name=params.log_name,
                             reuse_last_task_id=False)

    tasks = [parse_task(task) for task in params.tasks]
    neptune_logger_factory = NeptuneLoggerFactory(
        project=params.neptune_project,
        api_key=params.neptune_api_token,
        name=params.log_name,
    )

    data_dir = DataDir(data_dir=Path(params.data_dir))
    task_constructor = TaskConstructor(data_dir=data_dir)
    score_dir = Path(params.score_dir) if params.score_dir else None
    embeddings = np.load(params.embeddings_dir + '/embeddings.npy').astype('float32')
    clearml_logger = clearml_task.get_logger()
    clearml_logger.report_single_value(f'emb_dim', embeddings.shape[1])
    compute_metrics(embeddings, clearml_task=clearml_task)

    run_tasks(
        neptune_logger_factory=neptune_logger_factory,
        tasks=tasks,
        task_constructor=task_constructor,
        data_dir=data_dir,
        embeddings_dir=Path(params.embeddings_dir),
        num_workers=params.num_workers,
        accelerator=params.accelerator,
        devices=parse_devices(params.devices),
        score_dir=score_dir,
        disable_relevant_clients_check=params.disable_relevant_clients_check,
        clearml_task=clearml_task
    )


if __name__ == "__main__":

    parse = get_parser()
    params = parse.parse_args()
    main(params)
