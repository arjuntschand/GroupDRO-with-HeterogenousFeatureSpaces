"""EMBED bolster study — take the tuned result to paper level.

Builds on the proper tuned final (resnet18, 20k, lam=0.01/eta=3, 3 seeds) which showed
anchors_gdro best on all metrics (+5.7 worst-group vs ERM, resolves GDRO's overall drop).
This adds the rigor a reviewer wants:

  Study S (seeds):   erm/gdro/anchors_gdro at 2 MORE seeds (2024, 31337) -> 5 total,
                     tighter error bars + paired significance on the worst-group/overall gains.
  Study E (eta):     gdro & anchors_gdro across eta {1,2,5} (eta3 already done) -> shows
                     anchors help ACROSS eta, not a cherry-picked setting.
  Study A (ablation):anchors_shared (anchors, NO GDRO) + full (per-group enc) at tuned
                     settings -> clean per-component contribution, tuned (not the old
                     untuned ablation).

Every run also logs a view-dropout curve (missingness robustness). All resnet18,
num_workers=4 (resnet50 OOM-hangs on this box). Incremental results + DONE marker.

  nohup /opt/pytorch/bin/python run_embed_bolster.py > ~/bolster.log 2>&1 &
"""
import copy, json, os, time, traceback
import yaml
from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
HEAD = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROP = [0.0, 0.25, 0.5, 0.75]
OUT = "runs/embed_bolster"
LAM = 0.01
EPOCHS = 25
results = []
base = yaml.safe_load(open(BASE))


def run(tag, cell, seed, **ov):
    cfg = copy.deepcopy(base); cfg.update(CELLS[cell]); cfg.update(ov)
    cfg.update(dict(require_local_images=True, head_group_names=HEAD, dropout_eval_ps=DROP,
                    min_group_size=30, backbone="resnet18", epochs=EPOCHS, seed=seed,
                    num_workers=4, run_name=tag, run_dir=f"{OUT}/{tag}"))
    print(f"\n{'='*70}\n[{tag}] cell={cell} seed={seed} lam={cfg['lambda_fit']} "
          f"eta={cfg.get('groupdro_eta')} gdro={cfg['groupdro_enabled']} "
          f"per_view={cfg['per_view_encoders']}\n{'='*70}", flush=True)
    t0 = time.time()
    rec = {"tag": tag, "cell": cell, "seed": seed, "lambda": cfg["lambda_fit"],
           "eta": cfg.get("groupdro_eta"), "gdro": cfg["groupdro_enabled"],
           "per_view": cfg["per_view_encoders"]}
    try:
        out = train(cfg); rec.update(out); rec["minutes"] = round((time.time()-t0)/60, 1); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e); traceback.print_exc()
    results.append(rec)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    # Study S — extra seeds (combine with the 3 from the proper final -> 5 seeds)
    for seed in [2024, 31337]:
        run(f"S_erm_s{seed}", "erm_shared", seed)
        run(f"S_gdro_s{seed}", "gdro_shared", seed, groupdro_eta=3.0)
        run(f"S_agdro_s{seed}", "anchors_gdro_shared", seed, lambda_fit=LAM, lambda_sep=LAM, groupdro_eta=3.0)
    # Study E — eta sensitivity (seed 42; eta=3 already in the final)
    for eta in [1.0, 2.0, 5.0]:
        run(f"E_gdro_eta{eta}", "gdro_shared", 42, groupdro_eta=eta)
        run(f"E_agdro_eta{eta}", "anchors_gdro_shared", 42, lambda_fit=LAM, lambda_sep=LAM, groupdro_eta=eta)
    # Study A — tuned ablation: anchors-only, and per-group full
    for seed in [42, 1337]:
        run(f"A_anchors_s{seed}", "anchors_shared", seed, lambda_fit=LAM, lambda_sep=LAM)
        run(f"A_full_s{seed}", "full", seed, lambda_fit=LAM, lambda_sep=LAM, groupdro_eta=3.0)
    open(os.path.join(OUT, "DONE"), "w").write(str(time.time()))
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nBOLSTER COMPLETE: {n_ok}/{len(results)} runs ok")


if __name__ == "__main__":
    main()
