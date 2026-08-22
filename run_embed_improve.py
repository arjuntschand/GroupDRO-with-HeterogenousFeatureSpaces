"""EMBED improvement sweep — genuine upgrades (NOT seed-hunting), seed 42, to see
whether any legitimately raises the worst-group gap / absolute accuracy. Each upgrade
runs BOTH erm_shared (baseline) and anchors_gdro_shared (ours) so we measure the GAP,
not just an absolute that could move for either. Winner (if any beats current at seed
42) gets scaled to full seeds later. All resnet18 (resnet50 OOM-hangs on this box).

Levers explored: latent capacity, input resolution, longer training, bigger head,
Wasserstein anchor separation, and a combined "big" config.

  nohup /opt/pytorch/bin/python run_embed_improve.py > ~/improve.log 2>&1 &
"""
import copy, json, os, time, traceback
import yaml
from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
HEAD = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROP = [0.0, 0.5]
OUT = "runs/embed_improve"
LAM, ETA, SEED = 0.01, 3.0, 42
results = []
base = yaml.safe_load(open(BASE))

# name -> config overrides applied on top of the tuned baseline.
# NOTE: only image_size=224 is cached as .npy; other sizes re-decode DICOMs every batch
# (~28x slower), so we keep all upgrades at 224. Higher-res would need a 288px precache.
UPGRADES = {
    "ref":        {},                                              # current best config
    "latent256":  {"latent_dim": 256},
    "ep40":       {"epochs": 40},
    "head512":    {"head_hidden": 512},
    "sepw2":      {"sep_method": "w2", "sep_margin": 4.0},          # ours only (anchor sep)
    "bigcap":     {"latent_dim": 256, "epochs": 40, "head_hidden": 512},  # combined, still 224px
}


def run(tag, cell, **ov):
    cfg = copy.deepcopy(base); cfg.update(CELLS[cell]); cfg.update(ov)
    cfg.update(dict(require_local_images=True, head_group_names=HEAD, dropout_eval_ps=DROP,
                    min_group_size=30, backbone="resnet18", seed=SEED, num_workers=4,
                    run_name=tag, run_dir=f"{OUT}/{tag}"))
    cfg.setdefault("epochs", 25)
    print(f"\n[{tag}] cell={cell} latent={cfg['latent_dim']} img={cfg.get('image_size')} "
          f"ep={cfg['epochs']} head={cfg.get('head_hidden')} sep={cfg.get('sep_method')}", flush=True)
    t0 = time.time()
    rec = {"tag": tag, "cell": cell, "latent_dim": cfg["latent_dim"],
           "image_size": cfg.get("image_size"), "epochs": cfg["epochs"],
           "head_hidden": cfg.get("head_hidden"), "sep_method": cfg.get("sep_method")}
    try:
        out = train(cfg); rec.update(out); rec["minutes"] = round((time.time()-t0)/60, 1); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e); traceback.print_exc()
    results.append(rec); os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, ov in UPGRADES.items():
        # baseline (erm) at this upgrade — sepw2 only affects anchors so skip erm there
        if name != "sepw2":
            run(f"{name}_erm", "erm_shared", **ov)
        run(f"{name}_ours", "anchors_gdro_shared", lambda_fit=LAM, lambda_sep=LAM,
            groupdro_eta=ETA, **ov)
    open(os.path.join(OUT, "DONE"), "w").write(str(time.time()))
    print(f"IMPROVE DONE: {sum(1 for r in results if r.get('status')=='ok')}/{len(results)} ok")


if __name__ == "__main__":
    main()
