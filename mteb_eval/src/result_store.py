"""
result_store.py
Incremental CSV result storage. One row per
(model_name, task_name, pooling, layer_spec_name, mteb_score, **unsup_metrics).

Writes atomically (write to .tmp, then rename) to survive interruptions.
Thread-safe via a simple file lock. Supports in-place row refreshes when a
rerun adds new metric columns.
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

_lock = threading.Lock()

FIXED_COLUMNS = [
    "model_name",
    "task_name",
    "task_type",
    "pooling",
    "layer_spec",
    "mteb_score",
    "unsup_metrics_hash",
    "metric_error",
]


class ResultStore:
    """CSV store for experiment results with append and upsert support."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def append(self, row: Dict[str, Any]) -> None:
        with _lock:
            if self._header_written:
                existing_rows, existing_cols = self._read_all()
            else:
                existing_rows, existing_cols = [], []

            new_cols = [c for c in row.keys() if c not in existing_cols]
            all_cols = self._ordered_columns(existing_cols + new_cols)

            if not self._header_written or new_cols:
                self._write_all(all_cols, existing_rows + [row])
                self._header_written = True
            else:
                self._append_row(all_cols, row)

    def upsert(self, row: Dict[str, Any]) -> None:
        """Insert or replace a row keyed by model/task/pooling/layer_spec."""
        key_fields = ("model_name", "task_name", "pooling", "layer_spec")
        for field in key_fields:
            if field not in row:
                raise KeyError(f"upsert row missing required key field: {field}")

        with _lock:
            if self._header_written:
                existing_rows, existing_cols = self._read_all()
            else:
                existing_rows, existing_cols = [], []

            match_idx = None
            for idx, existing in enumerate(existing_rows):
                if all(existing.get(field) == str(row.get(field, "")) for field in key_fields):
                    match_idx = idx
                    break

            new_cols = [c for c in row.keys() if c not in existing_cols]
            all_cols = self._ordered_columns(existing_cols + new_cols)

            if match_idx is None and self._header_written and not new_cols:
                self._append_row(all_cols, row)
                return

            updated_rows = list(existing_rows)
            if match_idx is None:
                updated_rows.append(row)
            else:
                merged = dict(updated_rows[match_idx])
                merged.update(row)
                updated_rows[match_idx] = merged

            self._write_all(all_cols or list(row.keys()), updated_rows)
            self._header_written = True

    def read_df(self):
        import pandas as pd
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path)

    def get_row(
        self, model_name: str, task_name: str, pooling: str, layer_spec: str
    ) -> Dict[str, Any] | None:
        if not self.path.exists():
            return None
        with _lock:
            rows, _ = self._read_all()
        for r in rows:
            if (r.get("model_name") == model_name
                    and r.get("task_name") == task_name
                    and r.get("pooling") == pooling
                    and r.get("layer_spec") == layer_spec):
                return r
        return None

    def is_done(self, model_name: str, task_name: str,
                pooling: str, layer_spec: str) -> bool:
        return self.get_row(model_name, task_name, pooling, layer_spec) is not None

    def _read_all(self):
        rows, cols = [], []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or [])
                rows = list(reader)
        return rows, cols

    def _write_all(self, cols: List[str], rows: List[Dict]) -> None:
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in cols})
        os.replace(tmp, self.path)

    def _append_row(self, cols: List[str], row: Dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writerow({c: row.get(c, "") for c in cols})

    @staticmethod
    def _ordered_columns(cols: List[str]) -> List[str]:
        ordered = [c for c in FIXED_COLUMNS if c in cols]
        ordered.extend(c for c in cols if c not in ordered)
        return ordered
