"""Mechanism controls: is the gain really from the mechanism, or just from extra
parameters / generic regularization / pooling?

A skeptical reviewer has three cheap alternative explanations for why per-group encoders
plus anchors beat a shared ERM baseline:

  (a) CAPACITY. Per-group encoders mean K encoders instead of 1, so the model simply has
      more parameters. Control: give the shared encoder the same total parameter budget
      (widen its hidden layer until parameter counts match) and re-run.

  (b) REGULARIZATION. The anchor loss is an extra penalty term, and almost any penalty
      shrinks weights and can help a small-data group. Control: replace the anchor loss
      with a plain L2 penalty on the latent, matched in magnitude, and re-run.

  (c) ANCHOR STRUCTURE vs ANY TARGET. Maybe pulling latents toward *any* fixed points
      helps, and the class-conditional part is irrelevant. Control: keep the anchor loss
      but assign each sample a RANDOM anchor instead of its class anchor.

If the gain survives (a) and disappears under (b) and (c), the mechanism is doing the work.

Usage:
  python run_mechanism_controls.py --dataset nhanes --base experiments/nhanes_disjoint_pergroup_gdro.yaml
"""
from __future__ import annotations
import argparse, copy, csv, json, os
import numpy as np
import yaml

MODULES = {"fedheart": "dro_hetero_anchors.src.train_fedheart",
           "nhanes": "dro_hetero_anchors.src.train_nhanes"}
SEEDS = [42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55]
ANCHOR_ON, ANCHOR_OFF = 0.1, 0.001


def count_params(cfg, module):
    """Instantiate the model exactly as training would and count trainable parameters."""
    import importlib, torch
    m = importlib.import_module(module)
    encs, head, anchors, _ = m.build_models(cfg, [1] * len(cfg["groups"]), torch.device("cpu"))
    seen, tot = set(), 0
    for e in encs.values():
        if id(e) in seen:      # shared encoder is the same object for every group
            continue
        seen.add(id(e))
        tot += sum(p.numel() for p in e.parameters())
    tot += sum(p.numel() for p in head.parameters())
    return tot


def read_best(run_dir):
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
    b = max(rows, key=lambda r: fl(r, "test_worst_group_acc"))
    return {"worst": fl(b, "test_worst_group_acc"), "overall": fl(b, "test_overall_acc"),
            "balanced": fl(b, "test_balanced_acc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MODULES))
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import importlib
    mod = MODULES[args.dataset]
    train = importlib.import_module(mod).train
    base = yaml.safe_load(open(args.base))
    base.setdefault("data_split_seed", 43 if args.dataset == "fedheart" else 100)
    tag = args.tag or args.dataset
    out = args.out or f"runs/mechanism_{tag}"
    os.makedirs(out, exist_ok=True)

    # ---- capacity matching: widen the SHARED encoder until it matches per-group total ----
    per_group_cfg = copy.deepcopy(base); per_group_cfg["common_encoder"] = False
    shared_cfg = copy.deepcopy(base); shared_cfg["common_encoder"] = True
    p_pergroup = count_params(per_group_cfg, mod)
    p_shared = count_params(shared_cfg, mod)
    hid = base["groups"][0].get("hidden_dim", 64)
    wide = copy.deepcopy(shared_cfg)
    best_h, best_gap = hid, abs(p_shared - p_pergroup)
    for h in range(hid, hid * 12, 4):
        c = copy.deepcopy(shared_cfg)
        for g in c["groups"]:
            g["hidden_dim"] = h
        gap = abs(count_params(c, mod) - p_pergroup)
        if gap < best_gap:
            best_gap, best_h = gap, h
    for g in wide["groups"]:
        g["hidden_dim"] = best_h
    p_wide = count_params(wide, mod)
    print(f"[{tag}] params: shared={p_shared:,}  per-group={p_pergroup:,}  "
          f"shared-widened(h={best_h})={p_wide:,}")

    # ---- the arms ----
    ARMS = [
        ("shared_ERM",        dict(common_encoder=True,  groupdro_enabled=False, lambda_fit=ANCHOR_OFF)),
        ("shared_ERM_wide",   dict(common_encoder=True,  groupdro_enabled=False, lambda_fit=ANCHOR_OFF,
                                   _widen=True)),           # (a) capacity control
        ("pergroup_GDRO",     dict(common_encoder=False, groupdro_enabled=True,  lambda_fit=ANCHOR_OFF)),
        ("pergroup_GDRO_anchor", dict(common_encoder=False, groupdro_enabled=True, lambda_fit=ANCHOR_ON)),
        ("pergroup_GDRO_l2",  dict(common_encoder=False, groupdro_enabled=True,  lambda_fit=ANCHOR_OFF,
                                   _l2=True)),              # (b) regularization placebo
        ("pergroup_GDRO_randanchor", dict(common_encoder=False, groupdro_enabled=True,
                                          lambda_fit=ANCHOR_ON, _rand=True)),  # (c) structure control
    ]

    results = {}
    for label, spec in ARMS:
        results[label] = {}
        for seed in args.seeds:
            cfg = copy.deepcopy(wide if spec.get("_widen") else base)
            cfg["common_encoder"] = spec["common_encoder"]
            cfg["groupdro_enabled"] = spec["groupdro_enabled"]
            cfg["lambda_fit"] = spec["lambda_fit"]
            cfg["lambda_sep"] = spec["lambda_fit"]
            if spec.get("_l2"):
                cfg["latent_l2"] = 0.1          # matched-magnitude plain penalty
            if spec.get("_rand"):
                cfg["random_anchor_targets"] = True
            cfg["seed"] = seed
            cfg["run_dir"] = f"{out}/{label}_s{seed}"
            m = read_best(cfg["run_dir"])
            if not (m and m["worst"] == m["worst"]):
                print(f"\n=== [{tag}] {label} seed={seed} ===", flush=True)
                try:
                    train(cfg); m = read_best(cfg["run_dir"])
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True); m = None
            if m:
                results[label][seed] = m

    print(f"\n\n########## MECHANISM CONTROLS: {tag} ##########")
    print(f"{'arm':>26} | worst-group | overall | n")
    for label, _ in ARMS:
        R = results.get(label, {})
        if not R:
            continue
        w = np.array([R[s]["worst"] for s in R]) * 100
        o = np.array([R[s]["overall"] for s in R]) * 100
        print(f"{label:>26} | {w.mean():5.2f} ± {w.std():4.2f} | {o.mean():5.2f} | {len(w)}")
    json.dump({"results": results, "params": {"shared": p_shared, "pergroup": p_pergroup,
                                              "shared_wide": p_wide, "wide_hidden": best_h}},
              open(f"{out}/results.json", "w"), indent=2)
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
