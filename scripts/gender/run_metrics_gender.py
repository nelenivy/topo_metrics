import gc
import math
import sys
from time import time
from typing import Dict, Any, List, Tuple, Optional

import catboost
import numpy as np
import ripserplusplus as rpp
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

sys.path.append(
    "/home/dpetrovitch/dzagcoffee/topo_metrics/google-research/graph_embedding/metrics"
)

from metrics import (  # type: ignore
    rankme,
    coherence,
    pseudo_condition_number,
    alpha_req,
    stable_rank,
    ne_sum,
    self_clustering
)


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

from topology_gender import calculate_ph_dim

def compute_metrics(embeddings, selected_metrics=None, 
        n_samples=10, sample_fraction=1/20, verbose=0):    
    sample_size = max(1, int(sample_fraction * embeddings.shape[0]))

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

    sample_size = max(1, int(sample_fraction * embeddings.shape[0]))
    metrics, times = {}, {}

    for i in range(n_samples):
        sample = resample(embeddings, n_samples=sample_size,
                          replace=False, random_state=42 + i)
        u, s, _ = np.linalg.svd(sample, compute_uv=True, full_matrices=False)

        for name in selected_metrics:
            if name not in available_metrics:
                continue
            metric_fn = available_metrics[name]

            try:
                t0 = time()
                result = metric_fn(sample, u=u, s=s)
                elapsed = time() - t0

                if isinstance(result, dict):
                    for key, val in result.items():
                        metrics.setdefault(key, []).append(val)
                        times.setdefault(key, []).append(elapsed)
                else:
                    metrics.setdefault(name, []).append(result)
                    times.setdefault(name, []).append(elapsed)
            except Exception as e:
                print(f"⚠️  Ошибка при вычислении {name} на итерации {i}: {e}")

        gc.collect()

    mean_metrics = {f"metric_{k}": np.mean(v) for k, v in metrics.items()}
    std_metrics = {f"std_{k}": np.std(
        v) / (np.mean(v) + 1e-10) for k, v in metrics.items()}

    if verbose:
        print("\n📊 Средние значения метрик:")

        for k, v in mean_metrics.items():
            print(
                f"🧠 {k:35s} = {v:.4f} | ⏱ {np.mean(times.get(k.replace('metric_', ''), [0])):.4f} сек"
            )

    return {**mean_metrics, **std_metrics}


def eval_downstream(
    inf_test_embeddings: pd.DataFrame,
    targets: pd.DataFrame,
    col_id: str = "customer_id",
    target_col: str = "gender",
    downstream_type: str = "catboost"
) -> Tuple[float, float]:
    targets_df = targets.set_index(col_id)
    merged_df = inf_test_embeddings.merge(
        targets_df, how="inner", on=col_id).set_index(col_id)

    X = merged_df.drop(columns=[target_col]).values
    y = merged_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42)
    downstream_type = downstream_type.lower().strip()

    if downstream_type == "catboost":
        model = catboost.CatBoostClassifier(
            iterations=150, random_seed=42, verbose=0)
    elif downstream_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(
            128, 64), max_iter=300, random_state=42)
    elif downstream_type == "logreg":
        model = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42)
    else:
        raise ValueError(f"Неизвестный тип модели: {downstream_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_proba), X_train, X_test


def evaluate_one_emb(
    inf_test_embeddings: pd.DataFrame,
    targets: pd.DataFrame,
    selected_metrics: Optional[List[str]] = None,
    sample_fractions: Tuple[float, ...] = (1 / 20,),
    col_id: str = "customer_id",
    target_col: str = "gender",
    verbose: int = 0,
    n_samples: int = 10,
    downstream_type: str = "catboost"
) -> List[Dict[str, Any]]:
    embeddings_np = inf_test_embeddings.drop(
        columns=[col_id]
    ).to_numpy(dtype=np.float32)

    accuracy, auc, X_train, X_test = eval_downstream(
        inf_test_embeddings, targets, col_id, target_col, downstream_type
    )

    results = []
    for name, data in [('all', embeddings_np), ('train', X_train)]: #, ('test', X_test)]:
        for fraction in sample_fractions:
            metrics = compute_metrics(
                data, selected_metrics, n_samples, fraction, verbose
            )
            metrics = {f"{k}_{name}": v for k, v in metrics.items()}
            metrics.update(
                {"accuracy": accuracy, "roc_auc": auc, "sample_fraction": fraction}
            )
            results.append(metrics)

    return results
