"""
Compute intrinsic dimensionality with skdim.
"""

import time

import numpy as np
import skdim

FAST_GLOBAL_ESTIMATORS = ("TwoNN", "MLE", "MOM", "lPCA")
FAST_LOCAL_ESTIMATORS = ("MLE", "MOM")


def _dedupe_exact_rows(data: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return arr, 0
    unique = np.unique(np.ascontiguousarray(arr), axis=0)
    return unique, int(arr.shape[0] - unique.shape[0])


def compute_intrinsic_dim_global(data, estimator_names=None, verbose=False):
    """
    Compute global intrinsic dimension estimates using different estimators.

    Parameters
    ----------
    data : np.ndarray
        Input data for which to estimate the intrinsic dimension.
    estimator_names : list of str, optional
        List of estimator names from `skdim.id` to use. If None, defaults to ['MLE', 'lPCA', 'MOM'].
    verbose : bool, optional
        Whether to print progress messages.

    Returns
    -------
    results : dict
        Dictionary containing the computed intrinsic dimension for each estimator under keys
        like 'dim_{name}', and the computation time under keys like 'time_{name}'.
    """

    if estimator_names is None:
        estimator_names = ['MLE', 'lPCA', 'MOM']

    data, n_duplicate_rows = _dedupe_exact_rows(data)
    if n_duplicate_rows > 0:
        pass

    if data.shape[0] < 2:
        return {
            **{f"dim_{name}": float("nan") for name in estimator_names},
            **{f"time_{name}": 0.0 for name in estimator_names},
        }

    results = {}

    for name in estimator_names:
        if verbose:
            print(name)
        start_time = time.time()

        try:
            estimator = getattr(skdim.id, name)()
            dim = estimator.fit_transform(data.astype(np.float32))
            end_time = time.time()

            results[f'dim_{name}'] = dim
            results[f'time_{name}'] = end_time - start_time
            if verbose:
                print(f"Dimension: {float(np.asarray(dim)):.2f}, time taken: {end_time - start_time:.3f} seconds")

        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[f'time_{name}'] = time.time() - start_time

    return results


def compute_intrinsic_dim_local(data, estimator_names=None,
                                n_neighbors=100, n_jobs=1,
                                return_pointwise_dims=False, verbose=False):
    """
    Compute local intrinsic dimension estimates using different estimators.

    Parameters
    ----------
    data : np.ndarray
        Input data for which to estimate the intrinsic dimension.
    estimator_names : list of str, optional
        List of estimator names (as strings) from `skdim.id` to use. If None, will default to ['MLE', 'TLE', 'MOM'].
    n_neighbors : int, optional
        Number of neighbors to use for local estimators. Default is 100.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is 1.
    verbose : bool, optional
        Whether to print progress messages.

    Returns
    -------
    results : dict
        Dictionary for each estimator containing:
            - Summary statistics of the estimated local intrinsic dimensions:
                'dim_mean_{name}': mean
                'dim_std_{name}': standard deviation
                'dim_min_{name}': minimum
                'dim_max_{name}': maximum
                'dim_median_{name}': median
                'dim_q25_{name}': 25th percentile
                'dim_q75_{name}': 75th percentile
                'dim_q10_{name}': 10th percentile
                'dim_q90_{name}': 90th percentile
                'dim_q05_{name}': 5th percentile
                'dim_q95_{name}': 95th percentile
            - The total computation time for each estimator as 'time_{name}'.
    """

    if estimator_names is None:
        estimator_names = ['MLE', 'TLE', 'MOM']

    results = {}
    pointwise_dims = {}

    for name in estimator_names:
        if verbose:
            print(name)
        start_time = time.time()

        try:
            estimator = getattr(skdim.id, name)()
            dims = estimator.fit_transform_pw(
                data.astype(np.float32), n_neighbors=n_neighbors, n_jobs=n_jobs)
            end_time = time.time()

            results.update({
                f'dim_mean_{name}': dims.mean(),
                f'dim_std_{name}': dims.std(),
                f'dim_min_{name}': dims.min(),
                f'dim_max_{name}': dims.max(),
                f'dim_median_{name}': np.median(dims),
                f'dim_q25_{name}': np.percentile(dims, 25),
                f'dim_q75_{name}': np.percentile(dims, 75),
                f'dim_q10_{name}': np.percentile(dims, 10),
                f'dim_q90_{name}': np.percentile(dims, 90),
                f'dim_q05_{name}': np.percentile(dims, 5),
                f'dim_q95_{name}': np.percentile(dims, 95),
                f'time_{name}': end_time - start_time
            })
            if return_pointwise_dims:
                pointwise_dims[name] = dims

            if verbose:
                print(f"Dimension: {results[f'dim_mean_{name}']:.2f}, time taken: {end_time - start_time:.3f} seconds")

        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[f'time_{name}'] = time.time() - start_time

    if return_pointwise_dims:
        return results, pointwise_dims
    else:
        return results
