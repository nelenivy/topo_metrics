import gc
import math
import sys
from time import time

import catboost
import numpy as np
import pandas as pd
import ripserplusplus as rpp
from catboost import CatBoostClassifier, Pool
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

sys.path.append("../google-research/graph_embedding/metrics")

from metrics import (rankme,
                     coherence,
                     pseudo_condition_number,
                     alpha_req,
                     stable_rank,
                     ne_sum,
                     self_clustering)


def ripser_metric(embeddings, u=None, s=None):    
    diagrams = rpp.run("--format point-cloud", embeddings)
    persistence = {}
    #persistence["ripser_sum"] = 0
    # Compute condensed pairwise distances (1D array)
    distances = pdist(embeddings)
    # Convert to square distance matrix
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

from topology import calculate_ph_dim

def compute_metrics(embeddings_np, selected_metrics=None, 
        n_samples=10, sample_fraction=1/20, verbose=0):    
    sample_size = max(1, int(sample_fraction * embeddings_np.shape[0]))

    # Метрики
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

    for i in range(n_samples):
        sample = resample(embeddings_np, n_samples=sample_size,
                          replace=False, random_state=42 + i)
        u, s, _ = np.linalg.svd(sample, compute_uv=True, full_matrices=False)

        for metric_name in selected_metrics:
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

    averaged_metrics = {f"metric_{k}": np.mean(v) for k, v in metrics.items()}
    std_metrics = {f"std_{k}": np.std(
        v) / (np.mean(v) + 1e-10) for k, v in metrics.items()}

    averaged_times = {f"metric_{k}": np.mean(v) for k, v in times.items()}
    std_times = {k: np.std(v) for k, v in times.items()}

    if verbose:
        print("\n📊 Средние значения метрик и время вычисления:")
        for metric_name in averaged_metrics:
            metric_value = averaged_metrics[metric_name]
            metric_time = averaged_times.get(metric_name, None)
            print(
                f"🧠 {metric_name:30s} = {metric_value:.4f} | ⏱ {metric_time:.4f} сек")

    averaged_metrics = {**averaged_metrics, **std_metrics}
    return averaged_metrics


def eval_downstream(
    inf_test_embeddings,
    targets,
    col_id="customer_id",
    target_col="gender",
    downstream_type="catboost",  # 'catboost' | 'mlp' | 'logreg'
):
    # подготовка данных
    targets_df = targets.set_index(col_id)
    inf_test_df = inf_test_embeddings.merge(
        targets_df, how="inner", on=col_id).set_index(col_id)

    X = inf_test_df.drop(columns=[target_col]).values
    y = inf_test_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    downstream_type = downstream_type.lower().strip()

    if downstream_type == "catboost":
        model = catboost.CatBoostClassifier(
            iterations=150,
            random_seed=42,
            verbose=0,
        )
    elif downstream_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
            verbose=False,
        )
    elif downstream_type == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {downstream_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    return accuracy, auc_score


def evaluate_one_emb(inf_test_embeddings, targets, selected_metrics=None,
                     sample_fractions=tuple([1/20]),
                     col_id="customer_id", target_col='gender',
                     verbose=0, n_samples=10, downstream_type="catboost"):
    embeddings_np = inf_test_embeddings.drop(
        columns=[col_id]).to_numpy(dtype=np.float32)
    accuracy, auc = eval_downstream(inf_test_embeddings, targets,
                                    col_id=col_id, target_col=target_col,
                                    downstream_type=downstream_type)

    res = []

    for sample_fraction in sample_fractions:
        metrics = compute_metrics(embeddings_np, selected_metrics,
                                  sample_fraction=sample_fraction,
                                  verbose=verbose, n_samples=n_samples)
        metrics['accuracy'] = accuracy
        metrics['roc_auc'] = auc
        metrics['sample_fraction'] = sample_fraction
        # metrics['times'] = times

        res.append(metrics)

    return res
