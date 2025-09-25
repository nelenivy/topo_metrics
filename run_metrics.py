import sys
import catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
sys.path.append("google-research/graph_embedding/metrics")
import ripserplusplus as rpp
from sklearn.utils import resample
from time import time
from metrics import (rankme,
        coherence,
        pseudo_condition_number,
        alpha_req,
        stable_rank,
        ne_sum,
        self_clustering)
import gc
from sklearn.metrics import roc_auc_score

def ripser_metric(embeddings, u=None, s=None):    
    diagrams = rpp.run("--format point-cloud", embeddings)
    persistence = {}
    persistence["ripser_sum"] = 0

    for k in range(len(diagrams)):
        persistence_sum = sum([death - birth for birth, death in diagrams[k] if death > birth])
        persistence[f"ripser_sum_H{k}"] = persistence_sum
        persistence["ripser_sum"]+= persistence_sum

    return persistence


def compute_metrics(embeddings_np, selected_metrics=None, 
        n_samples=10, sample_fraction=1/20):    
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
        "ripser": ripser_metric
    }
    if selected_metrics is None:
        selected_metrics = list(available_metrics.keys())

    metrics = {name: [] for name in selected_metrics}
    times = {name: [] for name in selected_metrics}

    for i in range(n_samples):
        sample = resample(embeddings_np, n_samples=sample_size, replace=False, random_state=42 + i)
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

    #print(1)
    averaged_metrics = {f"metric_{k}": np.mean(v) for k, v in metrics.items()}
    #print(2)
    std_metrics = {f"std_{k}": np.std(v) / (np.mean(v) + 1e-10) for k, v in metrics.items()}
    
    averaged_times = {f"metric_{k}": np.mean(v) for k, v in times.items()}
    std_times = {k: np.std(v) for k, v in times.items()}

    print("\n📊 Средние значения метрик и время вычисления:")
    for metric_name in averaged_metrics:
        metric_value = averaged_metrics[metric_name]
        metric_time = averaged_times.get(metric_name, None)
        print(f"🧠 {metric_name:30s} = {metric_value:.4f} | ⏱ {metric_time:.4f} сек")
    #print(3)
    averaged_metrics = {**averaged_metrics, **std_metrics}
    #print(4)
    return averaged_metrics


def eval_downstream(inf_test_embeddings, targets, col_id="customer_id", target_col='gender'):
    #print(targets.shape)
    targets_df = targets.set_index(col_id)
    inf_test_df = inf_test_embeddings.merge(targets_df, how="inner", on=col_id).set_index(col_id)
    # print(inf_test_embeddings.shape)
    # print(list(inf_test_df.columns))

    X = inf_test_df.drop(columns=[target_col]).values
    y = inf_test_df[target_col].values
    # print(X)
    # print(y.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # print(X.shape, X_train.shape, X_test.shape)
    classifier = catboost.CatBoostClassifier(
        iterations=150,
        random_seed=42,
        verbose=0,
    )
    classifier.fit(X_train, y_train)    

    # Get feature importance scores
    # feature_importance = classifier.get_feature_importance(Pool(X_train, y_train))
    # feature_names = inf_test_df.drop(columns=[target_col]).columns
    
    # # Create a dataframe to examine
    # fi_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
    # fi_df = fi_df.sort_values(by="importance", ascending=False)
    
    # print(fi_df)
    accuracy = classifier.score(X_test, y_test)
    y_scores = classifier.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_scores[:, 1])    
    print(accuracy, auc_score)
    return accuracy, auc_score


def evaluate_one_emb(inf_test_embeddings, targets, selected_metrics=None, 
        sample_fractions=tuple([1/20]),
        col_id="customer_id", target_col='gender'):
    embeddings_np = inf_test_embeddings.drop(columns=[col_id]).to_numpy(dtype=np.float32)
    accuracy, auc = eval_downstream(inf_test_embeddings, targets, 
        col_id=col_id, target_col=target_col)

    res = []

    for sample_fraction in sample_fractions:
        metrics = compute_metrics(embeddings_np, selected_metrics, sample_fraction=sample_fraction)
        metrics['accuracy'] = accuracy
        metrics['roc_auc'] = auc
        metrics['sample_fraction'] = sample_fraction
        #metrics['times'] = times
        
        res.append(metrics)
    #print(res)
    
    return res
