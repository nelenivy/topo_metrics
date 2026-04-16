"""
GPU-friendly MTEB scoring for non-retrieval tasks using precomputed embeddings
from ``LayerEmbeddingStore`` / ``PooledLayerEmbeddingView``.

Uses each task's own metric helpers (``_calculate_scores``, ``_compute_metrics``, etc.)
where possible so main scores match ``mteb.evaluate`` for the same vectors.

Unsupported task types return ``None`` so the driver can fall back to ``mteb.evaluate``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks import (
    AbsTaskClassification,
    AbsTaskMultilabelClassification,
    AbsTaskPairClassification,
    AbsTaskSTS,
)
from mteb.results.task_result import TaskResult
from mteb.similarity_functions import compute_pairwise_similarity

from src.embedding_extractor import extract_embedding_matrix
from src.layer_spec import LayerSpec

logger = logging.getLogger(__name__)


def _finite_score_or_none(score: Any) -> Optional[float]:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _branch_split_dict(task: Any, hf_subset: str) -> Any:
    ds = task.dataset
    if hf_subset not in ds and hf_subset == "default":
        return ds
    return ds[hf_subset]


def _pick_split_name(task: Any, branch: Any) -> str:
    for s in getattr(task, "eval_splits", []) or []:
        if s in branch:
            return str(s)
    if "test" in branch:
        return "test"
    return list(branch.keys())[0]


def _rows_texts_sts_pair(ds: Any, col1: str, col2: str) -> Tuple[List[str], List[str]]:
    n = len(ds)
    t1, t2 = [], []
    for i in range(n):
        t1.append(str(ds[i][col1]).strip())
        t2.append(str(ds[i][col2]).strip())
    return t1, t2


def _pair_scores_torch(
    e1: np.ndarray,
    e2: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine sim, neg-Manhattan, neg-Euclidean (row-wise), matching MTEB sign conventions."""
    a = torch.from_numpy(e1.astype(np.float32)).to(device)
    b = torch.from_numpy(e2.astype(np.float32)).to(device)
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    cos = (a * b).sum(dim=-1).detach().cpu().numpy()
    man = -(a - b).abs().sum(dim=-1).detach().cpu().numpy()
    euc = -torch.linalg.norm(a - b, dim=-1).detach().cpu().numpy()
    return cos, man, euc


def _classification_proxy_device(device: str) -> torch.device:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return torch.device(device)
    return torch.device("cpu")


def _class_weight_tensor(
    y: np.ndarray,
    classes: np.ndarray,
    class_weight: Any,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if class_weight is None:
        return None

    if class_weight == "balanced":
        counts = np.array([(y == c).sum() for c in classes], dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        weights = float(len(y)) / (float(len(classes)) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if isinstance(class_weight, dict):
        weights = np.ones(len(classes), dtype=np.float32)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for label, weight in class_weight.items():
            if label in class_to_idx:
                weights[class_to_idx[label]] = float(weight)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    return None


def _predict_logistic_regression_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    device: str,
    evaluator_model: Any,
    seed: Optional[int] = None,
) -> Optional[np.ndarray]:
    if not isinstance(evaluator_model, LogisticRegression):
        return None
    dev = _classification_proxy_device(device)
    if dev.type != "cuda":
        return None

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    classes, y_idx = np.unique(y_train, return_inverse=True)
    if len(classes) < 2:
        return np.asarray([classes[0]] * len(X_test))

    max_iter = int(getattr(evaluator_model, "max_iter", 100) or 100)
    max_iter = max(20, min(max_iter, 150))
    c_value = float(getattr(evaluator_model, "C", 1.0) or 1.0)
    fit_intercept = bool(getattr(evaluator_model, "fit_intercept", True))
    class_weight = _class_weight_tensor(
        np.asarray(y_train), classes, getattr(evaluator_model, "class_weight", None), dev
    )
    reg_strength = 0.0 if getattr(evaluator_model, "penalty", "l2") == "none" else 1.0 / max(c_value, 1e-6)

    x_tr = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(dev)
    y_tr = torch.from_numpy(np.asarray(y_idx, dtype=np.int64)).to(dev)
    x_te = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(dev)

    model = torch.nn.Linear(x_tr.shape[1], len(classes), bias=fit_intercept).to(dev)
    torch.nn.init.zeros_(model.weight)
    if model.bias is not None:
        torch.nn.init.zeros_(model.bias)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=min(20, max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tr)
        loss = loss_fn(logits, y_tr)
        if reg_strength:
            loss = loss + 0.5 * reg_strength * model.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        pred_idx = model(x_te).argmax(dim=-1).detach().cpu().numpy()
    return classes[pred_idx]


def _predict_logistic_regression_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    evaluator_model: Any,
    seed: Optional[int] = None,
) -> np.ndarray:
    clf = clone(evaluator_model)
    if seed is not None and "random_state" in clf.get_params():
        clf.set_params(random_state=int(seed))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def sts_gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
) -> Optional[float]:
    if not isinstance(task, AbsTaskSTS):
        return None
    hf_subsets = list(task.hf_subsets) if task.hf_subsets is not None else list(task.dataset.keys())
    branch0 = _branch_split_dict(task, hf_subsets[0])
    split_name = _pick_split_name(task, branch0)
    nested: Dict[str, Dict[str, Any]] = {split_name: {}}
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")

    for hf_subset in hf_subsets:
        branch = _branch_split_dict(task, hf_subset)
        ds = branch[split_name]
        c1, c2 = task.column_names
        s1, s2 = _rows_texts_sts_pair(ds, c1, c2)
        E1 = extract_embedding_matrix(store, s1, spec, n_layers)
        E2 = extract_embedding_matrix(store, s2, spec, n_layers)
        cos_l, man_l, euc_l = _pair_scores_torch(E1, E2, dev)
        cos_scores = cos_l.tolist()
        manhattan = man_l.tolist()
        euclidean = euc_l.tolist()
        sim_np = compute_pairwise_similarity(encoder, E1, E2)
        if hasattr(sim_np, "tolist"):
            sim_list = sim_np.tolist()
        else:
            sim_list = list(np.asarray(sim_np).reshape(-1))

        scores_dict = {
            "cosine_scores": cos_scores,
            "manhattan_distances": manhattan,
            "euclidean_distances": euclidean,
            "similarity_scores": sim_list,
        }
        norm = [float(task._normalize(x)) for x in ds["score"]]
        metrics = task._calculate_scores(scores_dict, norm)
        subset_scores = dict(metrics)
        task._add_main_score(subset_scores)
        nested[split_name][hf_subset] = subset_scores

    tr = TaskResult.from_task_results(task, nested, evaluation_time=0.0)
    return float(tr.get_score(splits=[split_name]))


def pair_classification_gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
) -> Optional[float]:
    if not isinstance(task, AbsTaskPairClassification):
        return None
    hf_subsets = list(task.hf_subsets) if task.hf_subsets is not None else list(task.dataset.keys())
    branch0 = _branch_split_dict(task, hf_subsets[0])
    split_name = _pick_split_name(task, branch0)
    nested: Dict[str, Dict[str, Any]] = {split_name: {}}
    c1, c2 = task.input1_column_name, task.input2_column_name
    lab_col = task.label_column_name

    for hf_subset in hf_subsets:
        branch = _branch_split_dict(task, hf_subset)
        raw = branch[split_name]
        s1, s2 = _rows_texts_sts_pair(raw, c1, c2)
        E1 = extract_embedding_matrix(store, s1, spec, n_layers)
        E2 = extract_embedding_matrix(store, s2, spec, n_layers)
        sim_np = compute_pairwise_similarity(encoder, E1, E2)
        if hasattr(sim_np, "tolist"):
            sim_list = sim_np.tolist()
        else:
            sim_list = list(np.asarray(sim_np).reshape(-1))
        e1f, e2f = E1.astype(np.float32), E2.astype(np.float32)
        dot_scores = np.asarray([np.dot(e1f[i], e2f[i]) for i in range(len(e1f))])
        # Match ``PairClassificationEvaluator``: Manhattan/Euclidean are positive distances.
        distances = {
            "cosine_scores": (1.0 - paired_cosine_distances(E1, E2)).tolist(),
            "euclidean_distances": paired_euclidean_distances(E1, E2).tolist(),
            "manhattan_distances": paired_manhattan_distances(E1, E2).tolist(),
            "similarity_scores": sim_list,
            "dot_scores": dot_scores.tolist(),
        }
        labels = [int(raw[i][lab_col]) for i in range(len(raw))]
        subset_scores = task._compute_metrics(distances, labels)
        task._add_main_score(subset_scores)
        nested[split_name][hf_subset] = subset_scores

    tr = TaskResult.from_task_results(task, nested, evaluation_time=0.0)
    return float(tr.get_score(splits=[split_name]))


def classification_gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
) -> Optional[float]:
    if not isinstance(task, AbsTaskClassification) or isinstance(
        task, AbsTaskMultilabelClassification
    ):
        return None
    hf_subsets = list(task.hf_subsets) if task.hf_subsets is not None else list(task.dataset.keys())
    branch0 = _branch_split_dict(task, hf_subsets[0])
    split_name = _pick_split_name(task, branch0)
    nested: Dict[str, Dict[str, Any]] = {split_name: {}}
    in_col, y_col = task.input_column_name, task.label_column_name
    train_key = task.train_split

    for hf_subset in hf_subsets:
        branch = _branch_split_dict(task, hf_subset)
        if hasattr(branch, "select_columns"):
            ds = branch.select_columns([y_col, in_col])
        else:
            ds = branch
        train_split = ds[train_key]
        eval_split = ds[split_name]
        test_texts = [str(eval_split[i][in_col]).strip() for i in range(len(eval_split))]
        X_test = extract_embedding_matrix(store, test_texts, spec, n_layers)
        y_test = eval_split[y_col]

        scores = []
        idxs_state = None
        for i in range(task.n_experiments):
            train_ds, idxs_state = task._undersample_data(train_split, i, idxs_state)
            train_texts = [str(train_ds[j][in_col]).strip() for j in range(len(train_ds))]
            X_tr = extract_embedding_matrix(store, train_texts, spec, n_layers)
            y_tr = np.asarray(train_ds[y_col])
            y_pred = _predict_logistic_regression_gpu(
                X_tr,
                y_tr,
                X_test,
                device=device,
                evaluator_model=task.evaluator_model,
                seed=getattr(task, "seed", None),
            )
            if y_pred is None:
                y_pred = _predict_logistic_regression_sklearn(
                    X_tr,
                    y_tr,
                    X_test,
                    evaluator_model=task.evaluator_model,
                    seed=getattr(task, "seed", None),
                )
            scores.append(task._calculate_scores(y_test, y_pred))

        avg_scores: Dict[str, Any] = {
            k: (
                float(np.mean(values))
                if (values := [s[k] for s in scores if s[k] is not None])
                else float("nan")
            )
            for k in scores[0].keys()
        }
        full = {"scores_per_experiment": scores, **avg_scores}
        task._add_main_score(full)
        nested[split_name][hf_subset] = full

    tr = TaskResult.from_task_results(task, nested, evaluation_time=0.0)
    return float(tr.get_score(splits=[split_name]))


def multilabel_gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
) -> Optional[float]:
    if not isinstance(task, AbsTaskMultilabelClassification):
        return None
    from sklearn.preprocessing import MultiLabelBinarizer

    hf_subsets = list(task.hf_subsets) if task.hf_subsets is not None else list(task.dataset.keys())
    branch0 = _branch_split_dict(task, hf_subsets[0])
    split_name = _pick_split_name(task, branch0)
    nested: Dict[str, Dict[str, Any]] = {split_name: {}}
    in_col, y_col = task.input_column_name, task.label_column_name
    train_key = task.train_split

    for hf_subset in hf_subsets:
        branch = _branch_split_dict(task, hf_subset)
        if hasattr(branch, "select_columns"):
            ds = branch.select_columns([y_col, in_col])
        else:
            ds = branch
        train_split = ds[train_key]
        eval_split = ds[split_name]

        train_samples: List[List[int]] = []
        for _ in range(task.n_experiments):
            sample_indices, _ = task._undersample_data_indices(
                train_split[y_col], task.samples_per_label, None
            )
            train_samples.append(sample_indices)

        unique_train_indices = list(set(i for s in train_samples for i in s))
        utexts = [str(train_split[i][in_col]).strip() for i in unique_train_indices]
        X_unique = extract_embedding_matrix(store, utexts, spec, n_layers)
        emb_by_idx = {idx: X_unique[j] for j, idx in enumerate(unique_train_indices)}

        test_dataset = eval_split
        try:
            if len(test_dataset) > 2000:
                split_dataset = eval_split.train_test_split(
                    test_size=2000, seed=42, stratify_by_column="label"
                )
                test_dataset = split_dataset["test"]
        except ValueError:
            pass

        ttexts = [str(test_dataset[i][in_col]).strip() for i in range(len(test_dataset))]
        X_test = extract_embedding_matrix(store, ttexts, spec, n_layers)
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(test_dataset[y_col])

        from mteb.abstasks.multilabel_classification import _evaluate_classifier

        scores = []
        for sample_indices in train_samples:
            X_train = np.stack([emb_by_idx[idx] for idx in sample_indices])
            y_train_raw = train_split.select(sample_indices)[y_col]
            y_train = binarizer.transform(y_train_raw)
            y_pred, clf = _evaluate_classifier(
                X_train, y_train, X_test, task.evaluator_model
            )
            scores.append(task._calculate_scores(y_test, y_pred, X_test, clf))

        avg_scores = {k: float(np.mean([s[k] for s in scores])) for k in scores[0].keys()}
        full = {"scores_per_experiment": scores, **avg_scores}
        task._add_main_score(full)
        nested[split_name][hf_subset] = full

    tr = TaskResult.from_task_results(task, nested, evaluation_time=0.0)
    return float(tr.get_score(splits=[split_name]))


def non_retrieval_gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
) -> Optional[float]:
    """Return aggregate main score or ``None`` to trigger official ``mteb.evaluate``."""
    for fn in (
        sts_gpu_proxy_main_score,
        multilabel_gpu_proxy_main_score,
        classification_gpu_proxy_main_score,
        pair_classification_gpu_proxy_main_score,
    ):
        out = fn(task, store, spec, n_layers, device, encoder)
        if out is not None:
            score = _finite_score_or_none(out)
            if score is None:
                logger.warning(
                    "GPU proxy: non-finite %s score for %s; falling back to mteb.evaluate",
                    fn.__name__,
                    getattr(getattr(task, "metadata", None), "name", type(task).__name__),
                )
                return None
            return score
    return None
