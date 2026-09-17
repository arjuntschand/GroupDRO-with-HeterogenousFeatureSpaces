"""EMBED proper study — give the method a FAIR, TUNED shot on more data.

The first overnight run used blind defaults (lambda_fit=lambda_sep=1e-3, eta=1) on only
~8k samples with resnet18/20ep -> our method ~= baseline (77% vs REMIND's 80.7). Two
fixes: (a) MORE DATA (per-group 8000 download), (b) actually TUNE the anchor/GDRO
strength so the method exerts real force on the latent.

Stage 1 (sweep): baseline erm_shared + anchors_gdro_shared over a (lambda, eta) grid,
  1 seed. Answers: does ANY tuning make anchors+GDRO beat ERM on worst/tail group?
Stage 2 (final): erm_shared vs gdro_shared vs anchors_gdro_shared(best HP), N seeds,
  optionally resnet50, for the reported comparison.

Run on box:
  nohup /opt/pytorch/bin/python run_embed_proper.py --stage sweep > ~/proper_sweep.log 2>&1 &
  nohup /opt/pytorch/bin/python run_embed_proper.py --stage final --lam 0.03 --eta 3 \
      --backbone resnet50 --seeds 42 1337 7 > ~/proper_final.log 2>&1 &
"""
import argparse, copy, json, os, time, traceback
import yaml
from dro_hetero_anchors.src.train_embed import train
from run_embed_ablation import CELLS

BASE = "experiments/embed_base.yaml"
HEAD_NAMES = ["FFDM_CC+FFDM_MLO", "FFDM_CC+FFDM_MLO+CVIEW_CC+CVIEW_MLO"]
DROPOUT_PS = [0.0, 0.25, 0.5, 0.75]
# strength grid — deliberately spans well above the blind 1e-3 default
LAMS = [1e-2, 3e-2, 1e-1]
ETAS = [1.0, 3.0]

results = []


def run(base, out_dir, tag, cell, seed, epochs, backbone, **ov):
    global results
    cfg = copy.deepcopy(base); cfg.update(CELLS[cell]); cfg.update(ov)
    cfg.update(dict(require_local_images=True, head_group_names=HEAD_NAMES,
                    dropout_eval_ps=DROPOUT_PS, min_group_size=30, backbone=backbone,
                    epochs=epochs, seed=seed, num_workers=4,
                    run_name=tag, run_dir=f"{out_dir}/{tag}"))
    print(f"\n{'='*72}\n[{tag}] cell={cell} seed={seed} bb={backbone} ep={epochs} "
          f"lam={cfg['lambda_fit']} eta={cfg.get('groupdro_eta')} "
          f"per_view={cfg['per_view_encoders']} gdro={cfg['groupdro_enabled']}\n{'='*72}", flush=True)
    t0 = time.time()
    rec = {"tag": tag, "cell": cell, "seed": seed, "backbone": backbone, "epochs": epochs,
           "lambda": cfg["lambda_fit"], "eta": cfg.get("groupdro_eta"),
           "per_view": cfg["per_view_encoders"], "gdro": cfg["groupdro_enabled"]}
    try:
        out = train(cfg); rec.update(out); rec["minutes"] = round((time.time()-t0)/60, 1); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "FAILED"; rec["error"] = str(e); traceback.print_exc()
    results.append(rec)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "final"], required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7])
    ap.add_argument("--lam", type=float, default=0.03)   # final: chosen anchor strength
    ap.add_argument("--eta", type=float, default=3.0)    # final: chosen GDRO eta
    args = ap.parse_args()
    base = yaml.safe_load(open(BASE))

    if args.stage == "sweep":
        out = "runs/embed_proper_sweep"
        run(base, out, "erm_baseline", "erm_shared", 42, args.epochs, args.backbone)
        for lam in LAMS:
            for eta in ETAS:
                run(base, out, f"agdro_lam{lam}_eta{eta}", "anchors_gdro_shared", 42,
                    args.epochs, args.backbone,
                    lambda_fit=lam, lambda_sep=lam, groupdro_eta=eta)
        open(os.path.join(out, "DONE"), "w").write("sweep")
        print("SWEEP DONE")
    else:
        out = "runs/embed_proper_final"
        for seed in args.seeds:
            run(base, out, f"erm_s{seed}", "erm_shared", seed, args.epochs, args.backbone)
            run(base, out, f"gdro_s{seed}", "gdro_shared", seed, args.epochs, args.backbone,
                groupdro_eta=args.eta)
            run(base, out, f"agdro_s{seed}", "anchors_gdro_shared", seed, args.epochs, args.backbone,
                lambda_fit=args.lam, lambda_sep=args.lam, groupdro_eta=args.eta)
        open(os.path.join(out, "DONE"), "w").write("final")
        print("FINAL DONE")


if __name__ == "__main__":
    main()
