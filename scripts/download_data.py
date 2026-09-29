"""Download datasets from configs/datasets.yaml (T1.1).

Examples:
    # laptop: only Jülich trajectories + metadata (small)
    python scripts/download_data.py --groups must --roles trajectories metadata --only-source julich
    # A100: everything in must + should
    python scripts/download_data.py --root /data/crowdsafe/raw --groups must should
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdsafe.datasets.download import fetch_dataset, load_registry, select  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--groups", nargs="*", default=["must"])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--roles", nargs="*", default=None, help="e.g. trajectories metadata video images")
    ap.add_argument("--only-source", default=None, help="julich | kaggle | http")
    ap.add_argument("--report", type=Path, default=None, help="write a JSON summary here")
    args = ap.parse_args()

    reg = load_registry(args.config)
    chosen = select(reg, args.groups, args.names)
    if args.only_source:
        chosen = {k: v for k, v in chosen.items() if v["source"] == args.only_source}
    results = []
    for name, entry in chosen.items():
        print(f"--> {name}", flush=True)
        r = fetch_dataset(name, entry, args.root, args.roles, reg["max_single_download_gb"])
        print(f"    {r.status}" + (f": {r.error}" if r.error else ""),
              *(f"\n    {p}: {v['found']}/{v['expected']}" for p, v in r.expect.items()))
        results.append(asdict(r))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0 if all(r["status"] != "failed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
