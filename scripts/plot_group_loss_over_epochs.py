import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(run_dir: Path):
    # metrics.jsonl contains per-epoch entries from ResultsLogger
    log_path = run_dir / "metrics.jsonl"
    epochs = []
    g0_losses = []
    g1_losses = []
    if not log_path.exists():
        raise FileNotFoundError(f"Missing results file: {log_path}")
    with log_path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "epoch" not in rec:
                continue
            epochs.append(rec["epoch"])
            per_group_loss = rec.get("per_group_loss")
            if per_group_loss and len(per_group_loss) >= 2:
                g0_losses.append(per_group_loss[0])
                g1_losses.append(per_group_loss[1])
            else:
                g0_losses.append(None)
                g1_losses.append(None)
    return epochs, g0_losses, g1_losses


def plot_losses(epochs, g0_losses, g1_losses, title, out_path: Path):
    plt.figure(figsize=(7,4))
    plt.plot(epochs, g0_losses, label="MNIST loss", color="orange")
    plt.plot(epochs, g1_losses, label="USPS loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Mean cross-entropy loss (test)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to runs/<run_name> directory")
    ap.add_argument("--title", type=str, default="Per-group test loss over epochs")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    epochs, g0, g1 = load_results(run_dir)
    fig_dir = run_dir.parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"{run_dir.name}_group_loss_over_epochs.png"
    plot_losses(epochs, g0, g1, args.title, out_path)


if __name__ == "__main__":
    main()
