#!/usr/bin/env python3
"""
Autonomous Fed-Heart sweep: run baseline and GroupDRO configs with multiple seeds,
parse results, and optionally iterate with new configs.
"""
import subprocess
import sys
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
EXPERIMENTS = WORKSPACE / "experiments"
RUNS = WORKSPACE / "runs"


def run_config(config_name: str, seed: int) -> bool:
    """Run train_fedheart with given experiment config. Returns True if success."""
    config_path = EXPERIMENTS / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"[SKIP] Config not found: {config_path}")
        return False
    cmd = [
        sys.executable, "-m", "dro_hetero_anchors.src.train_fedheart",
        "--config", str(config_path),
    ]
    # Config has seed/run_dir; for multi-seed we need run_dir to differ. Override via env or new config.
    # For simplicity we use one config per (name, seed) by creating a temp config.
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = seed
    base_run = cfg.get("run_dir", "runs/unnamed")
    cfg["run_dir"] = f"{base_run}_seed{seed}"
    cfg["run_name"] = cfg.get("run_name", config_name) + f"_seed{seed}"
    tmp_path = EXPERIMENTS / f"_tmp_{config_name}_seed{seed}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "dro_hetero_anchors.src.train_fedheart", "--config", str(tmp_path)],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=600,
        )
        tmp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            print(f"[FAIL] {config_name} seed={seed}: {result.stderr[:500]}")
            return False
        print(f"[OK] {config_name} seed={seed}")
        return True
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        print(f"[ERR] {config_name} seed={seed}: {e}")
        return False


def get_best_metrics(run_dir: Path) -> dict | None:
    """Read metrics.jsonl and return best worst_group_acc and related."""
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None
    best = None
    best_acc = -1.0
    with open(metrics_file) as f:
        for line in f:
            d = json.loads(line)
            wg = d.get("test_worst_group_acc")
            if wg is not None and wg > best_acc:
                best_acc = wg
                best = {
                    "worst_group_acc": wg,
                    "balanced_acc": d.get("test_balanced_acc"),
                    "overall_acc": d.get("test_overall_acc"),
                    "epoch": d.get("epoch"),
                }
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["fedheart_baseline_strong", "fedheart_groupdro_strong"],
                    help="Config names (without .yaml)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339])
    ap.add_argument("--list-only", action="store_true", help="Only list run dirs and parsed results")
    args = ap.parse_args()

    if args.list_only:
        for run_dir in sorted(RUNS.iterdir()):
            if not run_dir.is_dir():
                continue
            m = get_best_metrics(run_dir)
            if m:
                print(f"{run_dir.name}: worst_group={m['worst_group_acc']:.4f} balanced={m['balanced_acc']:.4f} (epoch {m['epoch']})")
            else:
                print(f"{run_dir.name}: (no metrics)")
        return

    for config_name in args.configs:
        for seed in args.seeds:
            run_config(config_name, seed)

    print("\n--- Results ---")
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        m = get_best_metrics(run_dir)
        if m:
            print(f"{run_dir.name}: worst_group={m['worst_group_acc']:.4f} balanced={m['balanced_acc']:.4f} (epoch {m['epoch']})")


if __name__ == "__main__":
    main()
