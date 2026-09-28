"""Latent-space quality metrics from the saved test latents (runs/latent_points_*/<dataset>/<arm>_s<seed>.npz).

For each arm: mean distance of every point to the anchor of its own class (scale-normalised: divided by the
RMS norm of the latents), the fraction of points whose nearest anchor is their own class ('anchor accuracy'),
between-class centroid distance over mean within-class spread (separation ratio), the class silhouette, and
10-NN class purity. Arms without anchors use the class centroids in place of anchors (stated in the table).

  python compute_latent_metrics.py runs/latent_points_v4c/nhanes --arms no_anchors real_anchors
"""
import argparse, json, os
import numpy as np


def metrics(z, y, anchors=None):
    z = z.astype(np.float64); scale = np.sqrt((z ** 2).sum(1).mean())
    classes = sorted(set(y.tolist()))
    cent = np.stack([z[y == c].mean(0) for c in classes])
    A = anchors if anchors is not None else cent
    d_own = np.mean([np.linalg.norm(z[i] - A[classes.index(y[i])]) for i in range(len(z))]) / scale
    near = np.argmin(((z[:, None, :] - A[None, :, :]) ** 2).sum(-1), axis=1)
    anchor_acc = float(np.mean([classes[near[i]] == y[i] for i in range(len(z))]))
    within = np.mean([np.sqrt(((z[y == c] - cent[k]) ** 2).sum(1)).mean() for k, c in enumerate(classes)])
    between = np.mean([np.linalg.norm(cent[i] - cent[j]) for i in range(len(classes)) for j in range(i + 1, len(classes))])
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    n = min(len(z), 3000); rng = np.random.default_rng(0); idx = rng.choice(len(z), n, replace=False)
    sil = float(silhouette_score(z[idx], y[idx])) if len(classes) > 1 else float("nan")
    nn = NearestNeighbors(n_neighbors=11).fit(z); _, nb = nn.kneighbors(z[idx])
    purity = float(np.mean([(y[nb[i, 1:]] == y[idx[i]]).mean() for i in range(n)]))
    return dict(dist_to_own_anchor=float(d_own), anchor_accuracy=anchor_acc, separation_ratio=float(between / within),
                class_silhouette=sil, knn_purity=purity, latent_scale=float(scale), anchors_used=anchors is not None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--arms", nargs="+", default=["no_anchors", "real_anchors"])
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = {}
    for arm in a.arms:
        f = os.path.join(a.dir, f"{arm}_s{a.seed}.npz")
        if not os.path.exists(f):
            print("missing", f); continue
        d = np.load(f, allow_pickle=True); z, y = d["z"], d["y"]
        anc = d["anchor_m"] if ("anchors" in arm and "no_" not in arm and d["anchor_m"].any()) else None
        out[arm] = metrics(z, y, anc)
        out[arm]["per_group"] = {}
        for g in sorted(set(d["g"].tolist())):
            m = d["g"] == g
            if len(set(y[m].tolist())) > 1:
                out[arm]["per_group"][str(g)] = {k: v for k, v in metrics(z[m], y[m], anc).items() if k in ("dist_to_own_anchor", "anchor_accuracy", "separation_ratio", "knn_purity")}
        print(arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[arm].items() if k != "per_group"})
    json.dump(out, open(a.out or os.path.join(a.dir, "latent_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
