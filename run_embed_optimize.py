"""EMBED worst-group optimization pass — comprehensive sweep of every legitimate lever
that could raise worst-group accuracy, none properly tested before. Exploratory at 2
seeds (42,1337) to filter; any config that clearly beats ref -> validate at 10 seeds.

Levers:
  DRO objective/strength: weighted vs max (true min-max, worst-group-direct) vs logsumexp;
                          eta (DRO strength) 3/5/8; update_mode exp_smooth.
  Anchor separation:      lambda_sep higher.
  Learning rate:          1e-4, 2e-4 (base 5e-5).
  Optimizer:              SGD+nesterov momentum (base AdamW).
  LR schedule:            cosine, cosine+warmup.
  Combo:                  max objective + cosine.

All on anchors_gdro_shared, tuned base (lambda_fit=lambda_sep=0.01), resnet18/25ep,
existing 20k cached data. Compared vs the existing ERM baseline at the same seeds
(runs/embed_proper_final_results.json). Report ALL seeds — no cherry-picking.

  setsid /opt/pytorch/bin/python run_embed_optimize.py > ~/optimize.log 2>&1 < /dev/null &
"""
import copy, json, os, time, traceback
import yaml
from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
HEAD = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROP = [0.0, 0.5]
OUT = "runs/embed_optimize"
SEEDS = [42, 1337]
results = []
base = yaml.safe_load(open(BASE))

CONFIGS = {
    "ref":       {"groupdro_objective": "weighted",  "groupdro_eta": 3.0},
    # --- DRO objective / strength ---
    "max3":      {"groupdro_objective": "max",        "groupdro_eta": 3.0},
    "max5":      {"groupdro_objective": "max",        "groupdro_eta": 5.0},
    "lse3":      {"groupdro_objective": "logsumexp",  "groupdro_eta": 3.0},
    "expsmooth": {"groupdro_update_mode": "exp_smooth","groupdro_eta": 3.0},
    "eta8":      {"groupdro_objective": "weighted",   "groupdro_eta": 8.0},
    # --- anchor separation ---
    "sephi":     {"lambda_sep": 0.1,                  "groupdro_eta": 3.0},
    # --- learning rate ---
    "lr1e4":     {"lr": 1.0e-4,                       "groupdro_eta": 3.0},
    "lr2e4":     {"lr": 2.0e-4,                       "groupdro_eta": 3.0},
    # --- optimizer ---
    "sgd":       {"optimizer": "sgd", "lr": 1.0e-3, "momentum": 0.9, "groupdro_eta": 3.0},
    # --- LR schedule ---
    "cosine":    {"scheduler": "cosine",              "groupdro_eta": 3.0},
    "coswarm":   {"scheduler": "cosine_warmup", "epochs": 30, "groupdro_eta": 3.0},
    # --- combo: worst-group-direct objective + cosine schedule ---
    "max3_cos":  {"groupdro_objective": "max", "scheduler": "cosine", "groupdro_eta": 3.0},
}


def run(tag, seed, ov):
    cfg = copy.deepcopy(base); cfg.update(CELLS["anchors_gdro_shared"])
    cfg["lambda_fit"] = 0.01; cfg["lambda_sep"] = 0.01
    cfg.update(ov)
    cfg.update(dict(require_local_images=True, head_group_names=HEAD, dropout_eval_ps=DROP,
                    min_group_size=30, backbone="resnet18", seed=seed, num_workers=4,
                    run_name=tag, run_dir=f"{OUT}/{tag}"))
    cfg.setdefault("epochs", 25)
    print(f"\n[{tag}] seed={seed} obj={cfg.get('groupdro_objective')} eta={cfg.get('groupdro_eta')} "
          f"mode={cfg.get('groupdro_update_mode')} lr={cfg['lr']} opt={cfg.get('optimizer','adamw')} "
          f"sched={cfg.get('scheduler')} lam_sep={cfg['lambda_sep']}", flush=True)
    t0 = time.time()
    name = [k for k in CONFIGS if tag == f"{k}_s{seed}"][0]
    rec = {"tag": tag, "config": name, "seed": seed,
           "objective": cfg.get("groupdro_objective"), "eta": cfg.get("groupdro_eta"),
           "update_mode": cfg.get("groupdro_update_mode"), "lr": cfg["lr"],
           "optimizer": cfg.get("optimizer", "adamw"), "scheduler": cfg.get("scheduler"),
           "lambda_sep": cfg["lambda_sep"]}
    try:
        out = train(cfg); rec.update(out); rec["minutes"] = round((time.time()-t0)/60, 1); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e); traceback.print_exc()
    results.append(rec); os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, ov in CONFIGS.items():
        for seed in SEEDS:
            run(f"{name}_s{seed}", seed, ov)
    open(os.path.join(OUT, "DONE"), "w").write(str(time.time()))
    print(f"OPTIMIZE DONE: {sum(1 for r in results if r.get('status')=='ok')}/{len(results)} ok")


if __name__ == "__main__":
    main()
