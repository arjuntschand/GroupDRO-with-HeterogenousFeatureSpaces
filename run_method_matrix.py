"""ICML-grade method matrix for the tabular datasets, following Xenia's Step 5/6 protocol
(EMBED_Experiments_Description.docx) extended with regret optimization.

Methods (all share identical capacity; they differ only in the objective):
  1 ERM                     shared encoder,   no DRO,      no anchors
  2 GroupDRO (R*=0)         per-group enc,    DRO,         no anchors      <- standard baseline
  3 Regret-DRO              per-group enc,    regret-DRO,  no anchors      <- Xenia's regret
  4 Anchors only            per-group enc,    no DRO,      anchors
  5 Ours (anchors+GroupDRO) per-group enc,    DRO,         anchors
  6 Ours-regret             per-group enc,    regret-DRO,  anchors         <- full method
  7 Group-only              dedicated per-group models (this is what defines R*_g)

Rows 2/3/5/6 form the 2x2 (anchors x regret). Row 2 vs row 6 is the headline comparison.

Reported per Xenia Step 6: per-group accuracy, macro-F1, loss, R*_g, excess loss; overall
accuracy and macro-F1; worst-group raw loss (and which group); max excess loss (and which
group). Emits metrics_long.csv.

Usage:
  python run_method_matrix.py --dataset fedheart --base experiments/fedheart_exp_paper_hetagg_gdro.yaml \
      --rstar runs/rstar_fedheart.json --tag fedheart --moving-split
  python run_method_matrix.py --dataset nhanes --base experiments/nhanes_disjoint_pergroup_gdro.yaml \
      --rstar runs/rstar_nhanes_disjoint.json --tag nhanes_disjoint
"""
from __future__ import annotations
import argparse, ast, copy, csv, json, os
import numpy as np
import yaml

MODULES = {"fedheart": "dro_hetero_anchors.src.train_fedheart",
           "nhanes": "dro_hetero_anchors.src.train_nhanes"}
SEEDS = [42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55]
# ANCHOR_OFF is exactly 0: the trainer then skips the anchor terms (it was 0.001 before 2026-09-20)
ANCHOR_ON, ANCHOR_OFF = 0.1, 0.0

# (label, common_encoder, groupdro, use_regret, anchor_weight)
#
# Two different baselines are needed and they were being conflated. "ERM" here is a SHARED
# single encoder, so the ERM -> Ours gap bundles three separate changes: adding per-group
# encoders, adding anchors, and adding GroupDRO. Per-group encoding is itself one of the
# contributions (the synthetic overlap sweep shows it is worth up to +23.94 on its own at
# zero feature overlap), so folding it into the baseline gap both overstates what the anchors
# do and understates what the architecture does.
#
# PerGroupOnly isolates it: per-group encoders into a shared latent space and a shared head,
# with no anchors and no DRO. That makes the decomposition additive and readable:
#     ERM -> PerGroupOnly   = value of per-group encoders
#     PerGroupOnly -> GroupDRO = value of DRO reweighting
#     GroupDRO -> Ours_GDRO = value of the anchors
# It also matches what "ERM" means on EMBED, where XeniaEmbedModel always has per-group MLP_g,
# so EMBED's ERM row was already a PerGroupOnly row under a different name.
# The ablation is a full 2x2x2 over the three things the method actually adds:
#   encoder  in {shared, per-group}
#   DRO      in {off, on}
#   anchors  in {off, on}
# Only 4 of the 8 cells existed before, all of them per-group except ERM, which meant the
# encoder axis was never varied independently of the other two.
#
# Cell 3 (shared + DRO, no anchors) is the important omission. Our "GroupDRO" baseline is
# per-group + DRO, so it already contains the per-group architecture, which is one of our own
# contributions. GroupDRO as published (Sagawa et al.) reweights a SHARED model. Without cell 3
# we never showed the literature's actual baseline, and we could not separate "DRO helps" from
# "DRO helps once you have per-group encoders".
#
# (label, common_encoder, groupdro, use_regret, anchor_weight)
METHODS = [
    # --- encoder x DRO x anchors, all eight cells ---
    ("ERM",                 True,  False, False, ANCHOR_OFF),   # 1 shared,    -,   -
    ("PerGroupOnly",        False, False, False, ANCHOR_OFF),   # 2 per-group, -,   -
    ("Independent",         False, False, False, ANCHOR_OFF),   # per-group encoders AND heads: one model per group
    ("Shared_GDRO",         True,  True,  False, ANCHOR_OFF),   # 3 shared,    DRO, -
    ("GroupDRO",            False, True,  False, ANCHOR_OFF),   # 4 per-group, DRO, -
    ("Shared_Anchors",      True,  False, False, ANCHOR_ON),    # 5 shared,    -,   anchors
    ("AnchorsOnly",         False, False, False, ANCHOR_ON),    # 6 per-group, -,   anchors
    ("Shared_Anchors_GDRO", True,  True,  False, ANCHOR_ON),    # 7 shared,    DRO, anchors
    ("Ours_GDRO",           False, True,  False, ANCHOR_ON),    # 8 per-group, DRO, anchors
    # --- regret is a separate axis: it swaps raw loss for excess over R*_g in the DRO update ---
    ("RegretDRO",           False, True,  True,  ANCHOR_OFF),
    ("Ours_Regret",         False, True,  True,  ANCHOR_ON),
]


def parse_list(v):
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, str):
        try:
            return [float(x) for x in ast.literal_eval(v)]
        except Exception:
            return None
    return None


def extract(run_dir):
    """Best-worst-group epoch row -> all metrics we need."""
    f = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    def fl(r, k):
        try:
            return float(r.get(k, "") or "nan")
        except Exception:
            return float("nan")
    # Select the reported epoch on VALIDATION when a validation split exists. Selecting on
    # test, which is what this did before, makes every reported number a best-of-N on the test
    # set and biases it upward. Falls back to test only when no val column is present, and the
    # fallback is visible in the output rather than silent.
    sel_key = "val_worst_group_acc" if rows and rows[0].get("val_worst_group_acc") else "test_worst_group_acc"
    if sel_key == "test_worst_group_acc":
        globals().setdefault("_WARNED_TEST_SEL", False)
        if not globals()["_WARNED_TEST_SEL"]:
            print("  [warn] no validation split found; selecting the reported epoch on TEST",
                  flush=True)
            globals()["_WARNED_TEST_SEL"] = True
    best = max(rows, key=lambda r: fl(r, sel_key))
    out = {}
    for k in ("test_worst_group_acc", "test_overall_acc", "test_balanced_acc",
              "test_overall_macro_f1", "test_overall_auroc", "test_worst_group_bal_acc"):
        v = fl(best, k)
        if v == v:
            out[k[5:]] = v
    for k in ("test_per_group_acc", "test_per_group_loss", "test_per_group_f1",
              "test_per_group_auroc", "test_per_group_counts"):
        pl = parse_list(best.get(k))
        if pl:
            out[k[5:]] = pl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MODULES))
    ap.add_argument("--base", required=True)
    ap.add_argument("--rstar", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--moving-split", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to these arms (for sharding a matrix across processes)")
    args = ap.parse_args()

    import importlib
    train = importlib.import_module(MODULES[args.dataset]).train
    base = yaml.safe_load(open(args.base))
    if args.moving_split:
        base.pop("data_split_seed", None)
    else:
        base.setdefault("data_split_seed", 43 if args.dataset == "fedheart" else 100)
    rstar_d = json.load(open(args.rstar))["rstar"]
    rstar = [rstar_d[str(i)] if str(i) in rstar_d else rstar_d.get(i, 0.0)
             for i in range(len(rstar_d))]
    print(f"[{args.tag}] R*_g = " + ", ".join(f"g{i}:{r:.3f}" for i, r in enumerate(rstar)))

    out = args.out or f"runs/matrix_{args.tag}"
    os.makedirs(out, exist_ok=True)
    results, long_rows = {}, []

    for label, shared, gdro, regret, anch in METHODS:
        if args.methods and label not in args.methods:
            continue
        results[label] = {}
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["common_encoder"] = shared
            cfg["per_group_head"] = (label == "Independent")
            cfg["groupdro_enabled"] = gdro
            cfg["use_regret"] = regret
            if regret:
                cfg["optimal_losses"] = rstar
            cfg["lambda_fit"] = anch
            cfg["lambda_sep"] = anch
            cfg["seed"] = seed
            cfg["run_dir"] = f"{out}/{label}_s{seed}"
            m = extract(cfg["run_dir"])
            if not (m and m.get("worst_group_acc") == m.get("worst_group_acc")):
                print(f"\n=== [{args.tag}] {label} seed={seed} ===", flush=True)
                try:
                    train(cfg)
                    m = extract(cfg["run_dir"]) or {}
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True); m = {}
            results[label][seed] = m
            if m:
                print(f"  {label} s{seed}: worst={m.get('worst_group_acc',float('nan')):.4f} "
                      f"overall={m.get('overall_acc',float('nan')):.4f}", flush=True)
                pga = m.get("per_group_acc") or []
                pgl = m.get("per_group_loss") or []
                pgf = m.get("per_group_f1") or []
                pgc = m.get("per_group_counts") or []
                for gi in range(len(pga)):
                    L = pgl[gi] if gi < len(pgl) else float("nan")
                    R = rstar[gi] if gi < len(rstar) else float("nan")
                    long_rows.append({
                        "method": label, "seed": seed, "group": f"g{gi}",
                        "n": int(pgc[gi]) if gi < len(pgc) else "",
                        "accuracy": round(pga[gi], 5),
                        "macro_f1": round(pgf[gi], 5) if gi < len(pgf) else "",
                        "loss": round(L, 5) if L == L else "",
                        "R_star": round(R, 5) if R == R else "",
                        # signed, per Step 6. Clamping here floored our arms at zero while
                        # run_baselines_tabular reports signed, biasing max-excess our way.
                        "excess_loss": round(L - R, 5) if (L == L and R == R) else "",
                    })

    # metrics_long.csv (Xenia Step 6 schema)
    if long_rows:
        with open(f"{out}/metrics_long.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
            w.writeheader(); w.writerows(long_rows)
    json.dump({"results": results, "rstar": rstar}, open(f"{out}/results.json", "w"), indent=2)

    # console summary
    print(f"\n\n########## METHOD MATRIX: {args.tag} ##########")
    print(f"{'method':>14} | worst-grp | overall | macro-F1 | worst loss | max excess | n")
    for label, *_ in METHODS:
        if label not in results:
            continue
        R = results[label]
        def col(k):
            v = [R[s].get(k) for s in args.seeds if isinstance(R.get(s), dict)]
            v = [x for x in v if isinstance(x, (int, float)) and x == x]
            return np.mean(v) * 100 if v else float("nan")
        # worst loss + max excess computed per seed then averaged
        wl, mx = [], []
        for s in args.seeds:
            pgl = R.get(s, {}).get("per_group_loss")
            if pgl:
                wl.append(max(pgl))
                mx.append(max(l - rstar[i] for i, l in enumerate(pgl) if i < len(rstar)))
        n = sum(1 for s in args.seeds if R.get(s, {}).get("worst_group_acc") is not None)
        print(f"{label:>14} | {col('worst_group_acc'):8.2f} | {col('overall_acc'):7.2f} | "
              f"{col('overall_macro_f1'):8.2f} | {np.mean(wl) if wl else float('nan'):10.3f} | "
              f"{np.mean(mx) if mx else float('nan'):10.3f} | {n}")
    print(f"\nwrote {out}/results.json and {out}/metrics_long.csv")


if __name__ == "__main__":
    main()
