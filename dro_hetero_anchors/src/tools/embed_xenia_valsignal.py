"""Exploratory variant motivated by the diagnosed failure mode: the train-loss DRO
signal is fooled by tail memorization (tiny groups reach ~0 train loss, so the max
player abandons them). Drive the lambda update by VALIDATION loss instead.

Compares train-signal vs val-signal for GroupDRO and Ours, 3 seeds. Clearly labeled
as exploratory (deviates from Xenia's spec, which uses the train-loss EMA).

Usage: python -m dro_hetero_anchors.src.tools.embed_xenia_valsignal
"""
from __future__ import annotations
import numpy as np, torch
from ..train_embed_xenia import (
    load_group_tensors, patient_split, estimate_optimal_losses, train_one, evaluate,
    HEAD_GROUPS, GROUPS)

INDEX = "datasets/embed/index_xenia_offline.parquet"
CACHE = "datasets/embed/vit_cache"
SEEDS = [0, 1, 42]


def summarize(pg):
    accs = {g: v["acc"] for g, v in pg.items()}
    overall = np.average(list(accs.values()), weights=[pg[g]["n"] for g in accs])
    tail = np.mean([a for g, a in accs.items() if g not in HEAD_GROUPS])
    worst = min(accs.values())
    return overall, tail, worst


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_group_tensors(INDEX, CACHE)
    masks = patient_split(data, split_seed=0)
    groups = [g for g in GROUPS if g in data]

    print("method            signal | overall-wt |  tail  | worst  |  final lambda per group")
    for method in ["groupdro", "ours"]:
        for signal in ["train", "val"]:
            O, T, W, lams = [], [], [], []
            for seed in SEEDS:
                rstar = estimate_optimal_losses(data, masks, device, folds=5, seed=seed)
                model, info = train_one(method, data, masks, device, rstar, seed,
                                        epochs=20, dro_signal=signal, verbose=False)
                _, pg = evaluate(model, data, masks, "test", device, rstar)
                o, t, w = summarize(pg); O.append(o); T.append(t); W.append(w)
                lams.append(info["lambda"])
            lam_mean = np.mean(lams, axis=0)
            lam_str = " ".join(f"{g}:{l:.2f}" for g, l in zip(info["groups"], lam_mean))
            print(f"  {method:10s}  {signal:5s} | {np.mean(O):.3f}±{np.std(O):.3f} | "
                  f"{np.mean(T):.3f} | {np.mean(W):.3f} | {lam_str}")
    print("\n(train = Xenia's faithful spec; val = exploratory fix. tail/worst are the "
          "tail-robustness metrics of interest.)")


if __name__ == "__main__":
    main()
