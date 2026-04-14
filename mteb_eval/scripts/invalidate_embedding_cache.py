#!/usr/bin/env python3
"""
Remove LayerEmbeddingStore HDF5 caches so precompute runs again with new text keys.

Examples
--------
# Drop every precomputed store under a run (safest after corpus-string fixes):
python scripts/invalidate_embedding_cache.py --output-dir ./results/standard --all

# Show paths only:
python scripts/invalidate_embedding_cache.py --output-dir ./results/standard --all --dry-run

# Remove one exact store (same poolings order and n_layers as your eval script):
python scripts/invalidate_embedding_cache.py \\
  --output-dir ./results/standard \\
  --model bert-base-uncased --task SciFact --split test \\
  --poolings last_token mean cls --n-layers 13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache_manager import layer_store_hdf5_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="./results/standard",
        help="Run output directory (expects embedding_cache/ under it)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Delete all layer_store_*.h5 files under output-dir/embedding_cache",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be removed, do not delete",
    )
    p.add_argument("--model", help="With --task: model id (must match eval)")
    p.add_argument("--task", help="MTEB task name, e.g. SciFact")
    p.add_argument("--split", default="test", help="Split embedded in cache key")
    p.add_argument(
        "--poolings",
        nargs="+",
        help="Pooling names in the same order as the eval script",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        help="Layer count passed to LayerEmbeddingStore (hidden + 1 for BERT)",
    )
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    cache_dir = out / "embedding_cache"

    if args.all:
        if not cache_dir.is_dir():
            print(f"No directory {cache_dir}", file=sys.stderr)
            sys.exit(1)
        paths = sorted(cache_dir.glob("layer_store_*.h5"))
        if not paths:
            print(f"No layer_store_*.h5 under {cache_dir}")
            return
        for path in paths:
            print(("would remove " if args.dry_run else "removing ") + str(path))
            if not args.dry_run:
                path.unlink(missing_ok=True)
        return

    if args.model and args.task and args.poolings is not None and args.n_layers:
        path = layer_store_hdf5_path(
            cache_dir,
            args.model,
            args.task,
            args.split,
            list(args.poolings),
            int(args.n_layers),
        )
        print(("would remove " if args.dry_run else "removing ") + str(path))
        if not args.dry_run and path.exists():
            path.unlink()
        elif not args.dry_run:
            print(f"(missing) {path}", file=sys.stderr)
        return

    p.error("Use --all or pass --model, --task, --poolings ..., and --n-layers")


if __name__ == "__main__":
    main()
