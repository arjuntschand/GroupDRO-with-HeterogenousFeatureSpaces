"""EMBED head512 validation — does a bigger classifier head (head_hidden=512) genuinely
raise the worst-group gain? The improvement sweep flagged it at seed 42 (ours worst
74.0->76.1). This validates at the FULL 10 seeds for erm/gdro/anchors_gdro, same tuned
config (lam=0.01, eta=3, resnet18, 25 epochs, 224px cached) as the main result, so it is
a fair, apples-to-apples re-run with ONE legitimate change: head_hidden 256 -> 512.

Report ALL 10 seeds (no cherry-picking). Compare the head512 gap to the main +2.3.

  setsid /opt/pytorch/bin/python run_embed_head512.py > ~/head512.log 2>&1 < /dev/null &
"""
import copy, json, os, time, traceback
import yaml
from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
HEAD = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROP = [0.0, 0.5]
OUT = "runs/embed_head512"
LAM, ETA, EPOCHS, HEADH = 0.01, 3.0, 25, 512
SEEDS = [42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55]
results = []
base = yaml.safe_load(open(BASE))


def run(tag, cell, seed, **ov):
    cfg = copy.deepcopy(base); cfg.update(CELLS[cell]); cfg.update(ov)
    cfg.update(dict(require_local_images=True, head_group_names=HEAD, dropout_eval_ps=DROP,
                    min_group_size=30, backbone="resnet18", epochs=EPOCHS, seed=seed,
                    num_workers=4, head_hidden=HEADH, run_name=tag, run_dir=f"{OUT}/{tag}"))
    print(f"\n[{tag}] cell={cell} seed={seed} head_hidden={HEADH}", flush=True)
    t0 = time.time()
    rec = {"tag": tag, "cell": cell, "seed": seed, "head_hidden": HEADH}
    try:
        out = train(cfg); rec.update(out); rec["minutes"] = round((time.time()-t0)/60, 1); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e); traceback.print_exc()
    results.append(rec); os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    for seed in SEEDS:
        run(f"erm_s{seed}", "erm_shared", seed)
        run(f"gdro_s{seed}", "gdro_shared", seed, groupdro_eta=ETA)
        run(f"agdro_s{seed}", "anchors_gdro_shared", seed, lambda_fit=LAM, lambda_sep=LAM, groupdro_eta=ETA)
    open(os.path.join(OUT, "DONE"), "w").write(str(time.time()))
    print(f"HEAD512 DONE: {sum(1 for r in results if r.get('status')=='ok')}/{len(results)} ok")


if __name__ == "__main__":
    main()
