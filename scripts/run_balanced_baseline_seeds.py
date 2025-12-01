import subprocess, shutil, json, copy
from pathlib import Path
import argparse
import yaml

# Runs a balanced baseline config across multiple seeds, creating per-seed config variants.
# Example:
# python scripts/run_balanced_baseline_seeds.py \
#   --base experiments/mnist_usps_balanced_8k_5k_20e_baseline_lrdecay_lr0p1.yaml \
#   --seeds 1337 1338 1339 1340 1341

def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_yaml(obj, path: Path):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_seed(base_config: Path, seed: int, python_bin: str):
    cfg = load_yaml(base_config)
    prefix = cfg['run_name']
    seed_run_name = f"{prefix}_seed{seed}"
    cfg['seed'] = seed
    cfg['run_name'] = seed_run_name
    cfg['run_dir'] = f"runs/{seed_run_name}"
    seed_cfg_path = base_config.parent / f"{seed_run_name}.yaml"
    save_yaml(cfg, seed_cfg_path)
    print(f"[seed {seed}] running {seed_cfg_path}")
    cmd = [python_bin, '-m', 'dro_hetero_anchors.src.train', '--config', str(seed_cfg_path)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=str, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--python', type=str, default='python')
    args = ap.parse_args()

    base_cfg = Path(args.base)
    for seed in args.seeds:
        run_seed(base_cfg, seed, args.python)

if __name__ == '__main__':
    main()
