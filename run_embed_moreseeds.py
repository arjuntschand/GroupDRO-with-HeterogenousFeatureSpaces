"""EMBED extra seeds -> 10-seed total. Runs erm/gdro/anchors_gdro at 5 NEW seeds
(11,22,33,44,55) at the tuned config (lam=0.01, eta=3, resnet18). Combined with the
existing 5 seeds (42,1337,7 in embed_proper_final + 2024,31337 in embed_bolster S_)
this gives a 10-seed mean±std. We report ALL 10 (no cherry-picking).

  nohup /opt/pytorch/bin/python run_embed_moreseeds.py > ~/moreseeds.log 2>&1 &
"""
import copy, json, os, time, traceback
import yaml
from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
HEAD = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROP = [0.0, 0.25, 0.5, 0.75]
OUT = "runs/embed_10seed"
LAM, ETA, EPOCHS = 0.01, 3.0, 25
NEW_SEEDS = [11, 22, 33, 44, 55]
results = []
base = yaml.safe_load(open(BASE))


def run(tag, cell, seed, **ov):
    cfg = copy.deepcopy(base); cfg.update(CELLS[cell]); cfg.update(ov)
    cfg.update(dict(require_local_images=True, head_group_names=HEAD, dropout_eval_ps=DROP,
                    min_group_size=30, backbone="resnet18", epochs=EPOCHS, seed=seed,
                    num_workers=4, run_name=tag, run_dir=f"{OUT}/{tag}"))
    print(f"\n[{tag}] cell={cell} seed={seed} lam={cfg['lambda_fit']} eta={cfg.get('groupdro_eta')}", flush=True)
    t0 = time.time()
    rec = {"tag": tag, "cell": cell, "seed": seed, "lambda": cfg["lambda_fit"], "eta": cfg.get("groupdro_eta")}
    try:
        out = train(cfg); rec.update(out); rec["minutes"] = round((time.time()-t0)/60, 1); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e); traceback.print_exc()
    results.append(rec); os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    for seed in NEW_SEEDS:
        run(f"erm_s{seed}", "erm_shared", seed)
        run(f"gdro_s{seed}", "gdro_shared", seed, groupdro_eta=ETA)
        run(f"agdro_s{seed}", "anchors_gdro_shared", seed, lambda_fit=LAM, lambda_sep=LAM, groupdro_eta=ETA)
    open(os.path.join(OUT, "DONE"), "w").write(str(time.time()))
    print(f"MORESEEDS DONE: {sum(1 for r in results if r.get('status')=='ok')}/{len(results)} ok")


if __name__ == "__main__":
    main()
