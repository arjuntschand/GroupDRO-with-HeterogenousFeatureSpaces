"""EMBED full study — the overnight run. Benchmarks our method (per-group encoders +
Gaussian anchors + GroupDRO) on EMBED breast-density under modality-availability
heterogeneity, vs REMIND.

Three studies, all on the locally-cached image subset, with a PRINCIPLED semantic
head/tail split (head = complete-modality exams, tail = missing-view exams):

  A. Ablation      — all 7 method cells x seeds. Main table (mean +/- std), and since
                     every run also logs a view-dropout curve, this doubles as...
  C. Missingness   — accuracy vs test-time view-dropout p, per cell (robustness curves).
  B. Tail-starve   — vary how starved the tail is (tail_train_cap); measure the
                     ERM-vs-ours worst/tail gap. Shows gains grow as the tail thins.

Writes incremental results to runs/embed_study/results.json and touches
runs/embed_study/DONE when finished. Robust to per-run failures.

Run on the GPU box:
    nohup /opt/pytorch/bin/python run_embed_full_study.py > ~/study.log 2>&1 &
"""
import argparse, copy, json, os, time, traceback
import yaml

from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
OUT_DIR = "runs/embed_study"
RESULTS = os.path.join(OUT_DIR, "results.json")

# The two complete-modality combos are "head"; every incomplete combo is "tail".
HEAD_NAMES = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROPOUT_PS = [0.0, 0.25, 0.5, 0.75]

ALL_CELLS = ["erm_shared", "gdro_shared", "anchors_shared", "anchors_gdro_shared",
             "erm_pergroup", "gdro_pergroup", "full"]
STARVE_CAPS = [10, 50, 200, 100000]      # 100000 = effectively uncapped
STARVE_CELLS = ["erm_shared", "full"]

results = []


def _save():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)


def run_one(base, study, cell, seed, epochs, tag, **overrides):
    cfg = copy.deepcopy(base)
    cfg.update(CELLS[cell])
    cfg.update(overrides)
    cfg["seed"] = seed
    cfg["require_local_images"] = True
    cfg["head_group_names"] = HEAD_NAMES
    cfg["dropout_eval_ps"] = DROPOUT_PS
    cfg["epochs"] = epochs
    cfg["run_name"] = f"{study}_{tag}"
    cfg["run_dir"] = f"{OUT_DIR}/{study}_{tag}"
    print(f"\n{'='*72}\n[{study}] {tag}  (cell={cell} seed={seed} epochs={epochs} "
          f"per_view={cfg['per_view_encoders']} gdro={cfg['groupdro_enabled']} "
          f"anchors={cfg['lambda_fit']>0})\n{'='*72}", flush=True)
    t0 = time.time()
    rec = {"study": study, "cell": cell, "seed": seed, "tag": tag,
           "per_view": cfg["per_view_encoders"], "gdro": cfg["groupdro_enabled"],
           "anchors": cfg["lambda_fit"] > 0, "epochs": epochs, **overrides}
    try:
        out = train(cfg)
        rec.update(out)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e)
        traceback.print_exc()
    results.append(rec); _save()
    return rec


def main():
    global OUT_DIR, RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7])
    ap.add_argument("--starve-seeds", type=int, nargs="+", default=[42, 1337])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="1 cell/seed/epoch to validate")
    ap.add_argument("--only", choices=["A", "B", "all"], default="all")
    ap.add_argument("--starve-cells", nargs="+", default=STARVE_CELLS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    RESULTS = os.path.join(OUT_DIR, "results.json")

    with open(BASE) as f:
        base = yaml.safe_load(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    open(os.path.join(OUT_DIR, "STARTED"), "w").write(str(time.time()))

    if args.smoke:
        run_one(base, "smoke", "full", 42, 1, "full_s42")
        run_one(base, "smoke_starve", "erm_shared", 42, 1, "cap10_s42", tail_train_cap=10)
        open(os.path.join(OUT_DIR, "DONE"), "w").write("smoke")
        print("SMOKE DONE"); return

    # ---- Study A (+ C via dropout curves): full ablation ----
    if args.only in ("A", "all"):
        for cell in ALL_CELLS:
            for seed in args.seeds:
                run_one(base, "A_ablation", cell, seed, args.epochs, f"{cell}_s{seed}")

    # ---- Study B: tail-starvation curve ----
    if args.only in ("B", "all"):
        for cap in STARVE_CAPS:
            for cell in args.starve_cells:
                for seed in args.starve_seeds:
                    run_one(base, "B_starve", cell, seed, args.epochs,
                            f"{cell}_cap{cap}_s{seed}", tail_train_cap=cap)

    open(os.path.join(OUT_DIR, "DONE"), "w").write(str(time.time()))
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nSTUDY COMPLETE: {n_ok}/{len(results)} runs ok -> {RESULTS}")


if __name__ == "__main__":
    main()
