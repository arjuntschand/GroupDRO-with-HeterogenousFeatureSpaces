"""Fast TextCaps anchor sweep on CACHED frozen-ResNet features (full HF dataset).

Two modality groups: visual (cached 512-d ResNet feature) and text (OCR caption, CharCNN).
Each maps into a shared latent read by a shared head with class-conditional Gaussian anchors;
GroupDRO reweights the two modality groups. This is the frozen-backbone version of TextCaps
(consistent with EMBED's frozen ViT), so it runs on cached features — full data, fast.

Sweeps anchor arms (fit,sep). Usage (after extract_textcaps_features):
  python -m dro_hetero_anchors.src.train_textcaps_cached --cache datasets/textcaps/feat_cache \
      --arms 0.001,0.001 0.1,0.1 0.1,0 --seeds 1337 42 7
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss
from .model.groupdro import GroupDRO

VOCAB = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"-:;()")}
MAXLEN = 128


def encode_text(s):
    t = [VOCAB.get(c, 0) for c in str(s).lower()[:MAXLEN]]
    return t + [0] * (MAXLEN - len(t))


class TextEnc(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 1, 64, padding_idx=0)
        self.conv = nn.Conv1d(64, 128, 5, padding=2)
        self.fc = nn.Linear(128, latent)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = F.relu(self.conv(e)).max(dim=2).values
        return self.fc(h)


class VisEnc(nn.Module):
    def __init__(self, latent, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, latent), nn.ReLU(), nn.Linear(latent, latent))

    def forward(self, x):
        return self.net(x)


def load_split(cache, split):
    vis = np.load(f"{cache}/{split}_visual.f16.npy").astype(np.float32)
    meta = json.load(open(f"{cache}/{split}_meta.json"))
    caps = torch.tensor([encode_text(m["caption"]) for m in meta], dtype=torch.long)
    y = torch.tensor([m["label"] for m in meta], dtype=torch.long)
    return torch.tensor(vis), caps, y


def evaluate(vis_e, txt_e, head, Vx, Vy, Tx, Ty, device, num_classes):
    vis_e.eval(); txt_e.eval(); head.eval()
    accs = {}
    with torch.no_grad():
        for name, enc, X, Y in [("visual", vis_e, Vx, Vy), ("text", txt_e, Tx, Ty)]:
            logits = head(enc(X.to(device)))
            pred = logits.argmax(1).cpu()
            accs[name] = (pred == Y).float().mean().item()
    overall = np.mean(list(accs.values()))
    worst = min(accs.values())
    return worst, overall, accs


def train_one(data, lfit, lsep, seed, device, epochs=60, latent=64, lr=1e-3,
              num_classes=10, gamma=0.7, eta=0.1):
    dev = torch.device(device)
    Vx, Tx, y = data["train"]
    Vxt, Txt, yt = data["test"]
    torch.manual_seed(seed); np.random.seed(seed)
    vis_e = VisEnc(latent, Vx.shape[1]).to(device)
    txt_e = TextEnc(latent).to(device)
    head = nn.Linear(latent, num_classes).to(device)
    anchors = AnchorModule(num_classes, latent, eps=1e-4).to(device)
    params = list(vis_e.parameters()) + list(txt_e.parameters()) + list(head.parameters()) + list(anchors.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    gdro = GroupDRO(num_groups=2, eta=eta, gamma=gamma, update_mode="exp_smooth",
                    robust_objective="weighted", device=dev)

    Vx_d, Tx_d, y_d = Vx.to(device), Tx.to(device), y.to(device)
    n = len(y); bs = 256
    best_worst = 0.0
    for ep in range(epochs):
        vis_e.train(); txt_e.train(); head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            # group 0 = visual, group 1 = text (same samples, two modalities)
            zv = vis_e(Vx_d[idx]); zt = txt_e(Tx_d[idx]); yb = y_d[idx]
            lv = F.cross_entropy(head(zv), yb, reduction="mean")
            lt = F.cross_entropy(head(zt), yb, reduction="mean")
            # anchor losses on the union of both groups' latents
            z = torch.cat([zv, zt], 0); yy = torch.cat([yb, yb], 0)
            if lfit > 0 or lsep > 0:
                m_anc, S_anc, L_norm = anchors.forward()
                mom = per_class_batch_moments(z, yy, num_classes, 1e-4)
                l_fit = anchor_fit_loss(m_anc, S_anc, mom, 1e-4)
                l_sep = anchor_sep_loss(m_anc, S_anc, L_norm, head, num_classes, 8, dev,
                                        sep_method="classifier", margin=2.0, eps=1e-4)
            else:
                l_fit = torch.zeros((), device=device); l_sep = torch.zeros((), device=device)
            if ep >= 2:   # warmup: keep uniform q for first 2 epochs
                gdro.update_weights({0: lv.detach(), 1: lt.detach()},
                                    {0: int(len(yb)), 1: int(len(yb))})
            q = gdro.q
            task = q[0] * lv + q[1] * lt
            loss = task + lfit * l_fit + lsep * l_sep
            loss.backward(); opt.step()
        worst, overall, accs = evaluate(vis_e, txt_e, head, Vxt, yt, Txt, yt, device, num_classes)
        best_worst = max(best_worst, worst)
    return best_worst, overall, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="datasets/textcaps/feat_cache")
    ap.add_argument("--arms", nargs="+", default=["0.001,0.001", "0.1,0.1", "0.1,0", "0,0.1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 42, 7])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="runs/textcaps_cached")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(json.load(open(f"{args.cache}/classes.json")))
    Vx, Tx, y = load_split(args.cache, "train")
    Vxt, Txt, yt = load_split(args.cache, "val")
    print(f"full-data TextCaps (cached): train {len(y)}, val {len(yt)}, classes {num_classes}, "
          f"visual dim {Vx.shape[1]}")
    data = {"train": (Vx, Tx, y), "test": (Vxt, Txt, yt)}
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for a in args.arms:
        lf, ls = (float(x) for x in a.split(","))
        label = f"fit{lf}_sep{ls}"
        results[label] = {}
        for seed in args.seeds:
            w, o, accs = train_one(data, lf, ls, seed, device, epochs=args.epochs, num_classes=num_classes)
            results[label][seed] = {"worst": w, "overall": o, "per_group": accs}
            print(f"  [{label} s{seed}] worst={w:.4f} overall={o:.4f} vis={accs['visual']:.3f} txt={accs['text']:.3f}", flush=True)
    print("\n########## TEXTCAPS CACHED (FULL DATA) ANCHOR SWEEP ##########")
    print(f"{'arm':>16} | worst-group | overall | n")
    for label in results:
        w = [results[label][s]["worst"] for s in args.seeds]
        o = [results[label][s]["overall"] for s in args.seeds]
        print(f"{label:>16} | {np.mean(w)*100:5.2f} ± {np.std(w)*100:4.2f} | {np.mean(o)*100:5.2f} | {len(w)}")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=2)
    print(f"\nwrote {args.out}/results.json")


if __name__ == "__main__":
    main()
