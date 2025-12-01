import argparse
import subprocess
from pathlib import Path
import yaml

# Orchestrate 4 suites (each 5 seeds) and generate aggregated figures.
# Suites:
# 1) 8k/5k baseline
# 2) 8k/5k GroupDRO
# 3) 200/4k baseline
# 4) 200/4k GroupDRO
# Example:
# python scripts/run_all_suites.py --seeds 1337 1338 1339 1340 1341 \
#   --python .venv/bin/python

SUITES = [
    {
        'base': 'experiments/mnist_usps_skewsubset_8000_5000_flip_20e_baseline_lrdecay_lr0p1.yaml',
        'title': '8k MNIST vs 5k USPS (Baseline)',
        'colors': '#ff7f0e,#1f77b4', # orange, blue
    },
    {
        'base': 'experiments/mnist_usps_skewsubset_8000_5000_flip_20e_groupdro_eta0p30_gamma0p85_lrdecay_lr0p1.yaml',
        'title': '8k MNIST vs 5k USPS (GroupDRO)',
        'colors': '#2ca02c,#9467bd', # green, purple
    },
    {
        'base': 'experiments/mnist_usps_skewsubset_200_4000_flip_20e_baseline_lrdecay_lr0p1.yaml',
        'title': '200 MNIST vs 4k USPS (Baseline)',
        'colors': '#ff7f0e,#1f77b4',
    },
    {
        'base': 'experiments/mnist_usps_skewsubset_200_4000_flip_20e_groupdro_eta0p30_gamma0p85_lrdecay_lr0p1.yaml',
        'title': '200 MNIST vs 4k USPS (GroupDRO)',
        'colors': '#2ca02c,#9467bd',
    },
]


def load_yaml(p: Path):
    with open(p) as f:
        return yaml.safe_load(f)


def run_seeds(python_bin: str, base_cfg_path: Path, seeds):
    # reuse the existing runner
    cmd = [python_bin, 'scripts/run_balanced_baseline_seeds.py', '--base', str(base_cfg_path), '--seeds', *[str(s) for s in seeds]]
    subprocess.run(cmd, check=True)


def aggregate(python_bin: str, base_cfg_path: Path, colors: str, title: str, epoch: int = 20):
    cfg = load_yaml(base_cfg_path)
    prefix = cfg['run_name']
    out_name = f"figures/{prefix}_agg.png"
    cmd = [python_bin, 'scripts/aggregate_group_acc_plot.py', '--runs', 'runs', '--prefix', prefix,
           '--epoch', str(epoch), '--out', out_name, '--groups', '2', '--labels', 'MNIST,USPS', '--colors', colors,
           '--title', title, '--stat', 'se']
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--python', type=str, default='python')
    args = ap.parse_args()

    for suite in SUITES:
        base = Path(suite['base'])
        print(f"Running suite: {suite['title']} ({base})")
        run_seeds(args.python, base, args.seeds)
        print(f"Aggregating: {suite['title']}")
        aggregate(args.python, base, suite['colors'], suite['title'])

if __name__ == '__main__':
    main()
