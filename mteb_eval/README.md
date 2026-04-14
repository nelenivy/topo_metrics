# Unsupervised embedding quality evaluation on MTEB

This folder extends the **layer-aggregation** workflow: it scores many layer-wise
embedding configurations with **unsupervised** quality metrics and compares
those scores to **MTEB** test-split performance, without using task labels for
metric computation.

**Where the pieces come from**

- **Layer aggregation** — `AggregatedEncoder`, `LayerEncoder`, pooling, HDF5
  caching (`src/aggregated_encoder.py`, `src/cache_manager.py`,
  `src/strategies.py`, `src/utils.py`).
- **Spectral / intrinsic metrics** (topology-style pipeline) — subsampling,
  optional persistence (`ripser`) and PH-dimension hooks, wired in
  `src/unsup_metrics.py` (adapted from the same design as the original
  `run_metrics`-style evaluation loop).
- **Google Research embedding metrics** — `rankme`, `coherence`,
  `pseudo_condition_number`, `alpha_req`, `stable_rank`, `ne_sum`,
  `self_clustering` are implemented in **`src/metrics.py`** (same API as the
  public [google-research](https://github.com/google-research/google-research)
  unsupervised embedding metrics code).

By default, `unsup_metrics` imports those functions from **`src.metrics`**.
To use a different drop-in `metrics.py` instead, set **`GRAPH_METRICS_PATH`**
to the directory that contains it (it must export the same function names).

## Environment

Use the **`embs_aggr`** conda environment (recommended in this workspace):

```bash
conda activate embs_aggr
# interpreter: ~/.mlspace/envs/embs_aggr/bin/python
```

Install dependencies there if needed: `torch`, `transformers`, `mteb`, `scipy`,
`scikit-learn`, `datasets`, `h5py`, etc.

**Optional — `metric_ripser_*` columns:** persistence features need either
`pip install ripserplusplus` (fast, same as the original topology stack) or
`pip install ripser` (pure Python/C++ Ripser; used automatically if
`ripserplusplus` is missing). Use **`--ripser-maxdim 0`** to compute **H0 only**
(faster; omits H1-based `metric_ripser_*` features). Default is **`1`** (H0+H1).

## File structure

```
unsup_eval/
├── src/
│   ├── __init__.py
│   ├── aggregated_encoder.py   LayerEncoder + AggregatedEncoder (MTEB protocol)
│   ├── cache_manager.py        LayerEmbeddingStore (HDF5 per layer / pooling)
│   ├── embedding_extractor.py  extract_embedding_matrix, extract_retrieval_embeddings
│   ├── layer_spec.py           LayerSpec + build_layer_specs()
│   ├── metrics.py              Google Research–style unsupervised metrics
│   ├── result_store.py         Incremental CSV (master_results.csv)
│   ├── strategies.py           Weight normalization / aggregation helpers
│   ├── task_sets.py            CORE_TASKS, STANDARD_TASKS, FULL_BENCHMARK_NAME
│   ├── model_sets.py           CORE_MODELS, STANDARD_MODELS, FULL_MODELS
│   ├── pooling_rules.py        Skip invalid (model, pooling) pairs (e.g. cls on causal LMs)
│   ├── model_loading.py        HF load + tokenizer padding for encoders vs decoders
│   ├── unsup_metrics.py        compute_metrics, compute_metrics_retrieval
│   └── utils.py                Shared utilities
├── scripts/
│   ├── __init__.py
│   ├── run_unsup_eval.py       Main loop: cache embeddings → unsup metrics → MTEB
│   └── analyze_correlations.py Correlation + selection-quality analysis
└── README.md
```

## Quick start

From the **`unsup_eval`** directory (so `src` imports resolve):

```bash
cd /path/to/unsup_eval
conda activate embs_aggr
```

**Pooling** — `mean`, `cls`, and `last_token` are implemented in
`LayerEncoder.pool()` in `src/aggregated_encoder.py`. Causal LMs (GPT-style,
Llama, Mamba, …) should use **`mean`** or **`last_token`**; **`cls`** is for
BERT-like models. Loading uses `src/model_loading.py`: optional
`--torch-dtype float16|bfloat16|float32`, and **`--no-trust-remote-code`** if you
must disable custom model code (Mamba requires trust on).

### Sanity check (small model, one task, fast metrics)

```bash
conda activate embs_aggr
cd /path/to/unsup_eval
python scripts/run_unsup_eval.py \
  --models prajjwal1/bert-tiny \
  --tasks STSBenchmark \
  --poolings mean \
  --n-samples 2 \
  --min-sample-size 32 \
  --batch-size 8 \
  --output-dir ./results/sanity_unsup
```

Expect a `master_results.csv` under `./results/sanity_unsup/`. Warnings about
missing **`ripserplusplus`** only affect persistence-based `ripser_*` columns;
the spectral metrics from `src/metrics.py` still run.

**MTEB:** `AggregatedEncoder` sets `mteb_model_meta` to a full `mteb` **`ModelMeta`**
(built from the base HF model via `ModelMeta.from_hub`, then a unique
`aggregated/...` name). Older `SimpleNamespace` metadata breaks current MTEB.

### Larger runs

**Task sets** — `--task-set core|standard|full` (or override with `--tasks …`).

**Model sets** — if you **omit** `--models`, use **`--model-set core|standard|full`**
(lists in `src/model_sets.py`). Explicit `--models a b c` overrides the set.

We do **not** add E5/BGE instruction prefixes or chat templates by default (plain
text encoding for a consistent baseline).

**Invalid (model, pooling) pairs** — e.g. **`cls`** on causal LMs — are **skipped
by name heuristics** (`src/pooling_rules.py`) with a log line, not by try/except.

```bash
# Core tasks + predefined core model list, three poolings (cls skipped for decoders)
python scripts/run_unsup_eval.py \
  --model-set core \
  --task-set core \
  --poolings mean cls last_token \
  --output-dir ./results/unsup_eval

# Explicit models (same as before)
python scripts/run_unsup_eval.py \
  --models bert-base-uncased sentence-transformers/all-MiniLM-L6-v2 \
  --task-set core \
  --poolings mean cls last_token \
  --output-dir ./results/unsup_eval

# Full MTEB-v2 benchmark, one model
python scripts/run_unsup_eval.py \
  --models sentence-transformers/all-mpnet-base-v2 \
  --task-set full \
  --poolings mean \
  --output-dir ./results/unsup_eval
```

**Where time goes:** Most wall time is usually **GPU** (forward passes for layer
embeddings + MTEB’s own encoding). Unsup metrics and `ripser` are often smaller
unless `n_samples` and embedding dimension are large. **Speed-ups:** run
**separate processes** per model or per GPU (`CUDA_VISIBLE_DEVICES=0` / `1`),
or increase `--batch-size` if memory allows; CPU-side parallelism would require
code changes (e.g. multiprocessing tasks), not enabled by default.

### Analyse results

**Recommended:** give **each evaluation run its own `--output-dir`** (e.g.
`./results/exp_20260413/`) so `master_results.csv` is not mixed across reruns.

By default, `analyze_correlations.py` picks the **newest**
`./results/**/master_results.csv` (by mtime) and writes CSVs next to it under
**`<that_run>/correlations/`** (omit `--output-dir` unless you want a custom path):

```bash
python scripts/analyze_correlations.py \
  --search-root ./results \
  --sign-estimation per_task
```

Or analyse one run explicitly (outputs go to `./results/my_run/correlations/`):

```bash
python scripts/analyze_correlations.py \
  --results-csv ./results/my_run/master_results.csv \
  --sign-estimation per_task
```

Override the analysis folder if needed:

```bash
python scripts/analyze_correlations.py \
  --results-csv ./results/my_run/master_results.csv \
  --output-dir ./results/shared_correlations \
  --sign-estimation per_task
```

**Outputs** (under `correlations/`): `correlation/` holds one CSV per view
(pooled **all**, **per_task** / **per_model** Fisher means, **per_dataset** per
MTEB task, **per_task_type_pooled** and **per_task_type_fisher_mean** for
task-type slices). `selection_quality.csv` is the full table; `summary/` has
**overall**, **by_task_type**, and **by_dataset** aggregates of selection gaps.

---

## What gets evaluated

### Layer specifications (per model with L layers)

| Spec name | Type | Description |
|-----------|------|-------------|
| `layer_0` … `layer_{L-1}` | single | Individual hidden layers |
| `last_1`, `last_2`, `last_4`, `last_8` | last_k | Equal-weight mean of last k layers (each listed only if `k ≤ L`) |
| `all_mean` | all_mean | Equal-weight mean of all L layers |

Total per (model, task, pooling): **`L` single-layer specs + up to four `last_k` specs + `all_mean`**, i.e. `L + |{1,2,4,8} ∩ [1,L]| + 1` (for `L ≥ 8` that is **`L + 5`**).

### Retrieval tasks — three metric variants

| Variant | Description |
|---------|-------------|
| `metric_*_corpus` | Unsup metrics on corpus embeddings only |
| `metric_*_queries` | Unsup metrics on query embeddings only |
| `metric_*_combined` | Unsup metrics on queries + corpus sample (≤5000) |

Each variant is a separate column in `master_results.csv` and feeds
correlation / selection-quality outputs.

### Poolings

| Name | Method |
|------|--------|
| `mean` | Attention-masked token mean |
| `cls` | [CLS] token |
| `last_token` | Last non-padding token (`attention_mask.sum(-1) - 1`) |

---

## Outputs

### `master_results.csv`

One row per `(model_name, task_name, task_type, pooling, layer_spec)`.

Fixed columns: `model_name, task_name, task_type, pooling, layer_spec, mteb_score`

Variable columns: `metric_<name>` and `std_<name>` for each metric.
For retrieval tasks these are further suffixed `_corpus`, `_queries`, `_combined`.

### `correlation_results.csv`

| Column | Description |
|--------|-------------|
| `metric` | Metric column name |
| `granularity` | `all` / `per_task` / `per_model` |
| `spearman_r` | Spearman correlation with mteb_score |
| `spearman_p` | p-value (only for `all` granularity) |
| `pearson_r` | Pearson correlation (only for `all`) |
| `n` | Number of data points / groups |

### `selection_quality.csv`

| Column | Description |
|--------|-------------|
| `oracle_score` | max MTEB score among all configs |
| `selected_score` | MTEB score of config chosen by unsup metric |
| `gap` | oracle - selected |
| `selected_rank` | Rank of selected config (1 = best) |
| `n_configs` | How many configs were available |

### `metric_summary.csv`

Aggregation of `selection_quality.csv` per metric:
`mean_gap, median_gap, top1_accuracy, top3_accuracy, mean_selected_rank`

---

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | HuggingFace model names |
| `--task-set` | `core` | `core` / `standard` / `full` |
| `--tasks` | None | Explicit task names (overrides `--task-set`) |
| `--poolings` | `mean` | `mean` `cls` `last_token` |
| `--min-sample-size` | `100` | Minimum subsample size for metrics |
| `--sample-fraction` | `0.05` | Fraction of N per subsample |
| `--n-samples` | `10` | Number of subsamples to average |
| `--include-ph-dim` | off | Also compute PH dimension (slow) |
| `--overwrite` | off | Re-run already-computed configs |

---

## Incremental / resumable

The evaluation is resumable. Re-run the same command without `--overwrite`;
already-computed `(model, task, pooling, layer_spec)` rows are skipped.

HDF5 embedding stores are cached per `(model, task, pooling)` — changing layer
specs does **not** re-encode sentences for that triple.
