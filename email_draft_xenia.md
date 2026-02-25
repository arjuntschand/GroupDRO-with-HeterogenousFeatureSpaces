Hi Xenia,

I hope you're doing well and had a good weekend. I wanted to send an update on the TextCaps experimentation. I finished a new round of runs with the updated GroupDRO and baseline settings we discussed, and added the 3-group setup. All six runs are done. Below are the results in table form, including per-group and per-class accuracies and the final GroupDRO weights for the last epoch.

---

**Summary (epoch 50)**

| Run | Test acc | Worst-group acc | Per-group acc (visual, text, [combined]) |
|-----|----------|-----------------|------------------------------------------|
| 4-class 2-group baseline | 67.9% | 60.3% | 75.4%, 60.3% |
| 4-class 2-group GroupDRO | 67.4% | 60.9% | 73.9%, 60.9% |
| 10-class 2-group baseline | 67.8% | 60.6% | 74.9%, 60.6% |
| 10-class 2-group GroupDRO | 61.9% | 60.4% | 63.4%, 60.4% |
| 4-class 3-group baseline | 70.1% | 59.7% | 74.9%, 59.7%, 75.8% |
| 4-class 3-group GroupDRO | 65.6% | 59.7% | 71.9%, 59.7%, 65.3% |

**GroupDRO final weights (epoch 50)** — π was (0.5, 0.5) for 2-group and (⅓, ⅓, ⅓) for 3-group.

| Run | Final q (visual, text, [combined]) |
|-----|------------------------------------|
| 4-class 2-group GroupDRO | 0.003, 0.997 |
| 10-class 2-group GroupDRO | 0.0014, 0.9986 |
| 4-class 3-group GroupDRO | 0.0009, 0.993, 0.006 |

So in all GroupDRO runs the weights collapse to the worst (text) group.

---

**Per-class accuracy by group (epoch 50)**  
Classes: 0 = Bottle, 1 = Car, 2 = Food, 3 = Book (4-class). For 10-class, 0–9 are the top-10 class indices.

**4-class 2-group baseline**

| Class | Visual | Text |
|-------|--------|------|
| Bottle (0) | 55.2% | 50.0% |
| Car (1) | 93.2% | 50.8% |
| Food (2) | 51.8% | 33.6% |
| Book (3) | 93.5% | 81.9% |

**4-class 2-group GroupDRO**

| Class | Visual | Text |
|-------|--------|------|
| Bottle (0) | 53.4% | 43.1% |
| Car (1) | 91.5% | 52.5% |
| Food (2) | 45.5% | 34.5% |
| Book (3) | 94.4% | 86.1% |

**10-class 2-group baseline**

| Class | Visual | Text |
|-------|--------|------|
| 0 | 73.7% | 60.7% |
| 1 | 77.6% | 58.8% |
| 2 | 64.5% | 35.5% |
| 3 | 77.6% | 60.5% |
| 4 | 85.7% | 68.1% |
| 5 | 79.6% | 72.8% |
| 6 | 69.5% | 48.8% |
| 7 | 59.7% | 61.7% |
| 8 | 86.5% | 70.3% |
| 9 | 85.9% | 77.8% |

**10-class 2-group GroupDRO**

| Class | Visual | Text |
|-------|--------|------|
| 0 | 89.2% | 66.5% |
| 1 | 58.8% | 64.7% |
| 2 | 42.0% | 35.5% |
| 3 | 65.3% | 53.1% |
| 4 | 60.5% | 69.7% |
| 5 | 69.4% | 76.2% |
| 6 | 51.8% | 46.3% |
| 7 | 5.4% | 43.6% |
| 8 | 60.4% | 64.0% |
| 9 | 79.8% | 79.8% |

**4-class 3-group baseline**

| Class | Visual | Text | Combined |
|-------|--------|------|----------|
| Bottle (0) | 55.2% | 54.3% | 59.5% |
| Car (1) | 91.5% | 37.3% | 86.4% |
| Food (2) | 52.7% | 31.8% | 51.8% |
| Book (3) | 92.1% | 82.9% | 93.9% |

**4-class 3-group GroupDRO**

| Class | Visual | Text | Combined |
|-------|--------|------|----------|
| Bottle (0) | 52.6% | 52.6% | 53.4% |
| Car (1) | 88.1% | 39.0% | 52.5% |
| Food (2) | 46.4% | 33.6% | 37.3% |
| Book (3) | 90.7% | 82.4% | 89.4% |

---

**Changes we made in this round**

1. **Group weight initialization and baseline** — We no longer initialize group weights equally. Each group’s weight is set proportional to that group’s sample size (π_g = #samples in group g / #total samples). As you’d said: in the training set groups appear with a distribution π, and it’s better if batches preserve this distribution (stratified batches). The baseline now also uses these π weights for the per-group loss (π-weighted baseline).

2. **GroupDRO weights and warmup** — GroupDRO weights can go to 1 with no min/max clamp. We use only 3 epochs of warmup before the GroupDRO updates start.

3. **KL penalty** — We added a penalty in the loss: λ_KL × KL(current q || π), with π the initial distribution from group sizes. As you’d described: we learn weights λ so the model isn’t tied to π, but a good initialization for λ is π, and the KL penalty keeps us from diverging too much from π. We control this with a coefficient (e.g. 0.1).

4. **Per-class accuracies per group** — We now track and print per-class accuracy for each group in the terminal and log it with the rest of the metrics.

5. **Logging and stored results** — At the start of training we print and log total samples, per-group sample counts, per-class sample counts, and all hyperparameters. The same information is written to the results file (e.g. metrics.jsonl / metrics.csv) so each run is fully documented.

Hyperparameters: 50 epochs, batch size 32, Adam lr 1e-3, weight decay 5e-4, head dropout 0.5, visual encoder LR scale 0.01. For GroupDRO: eta 0.01, gamma 0.5, 3-epoch warmup, KL penalty coefficient 0.1, no min/max on group weights.

---

So far GroupDRO helps a bit in 4-class 2-group (60.3% → 60.9% worst-group) but in 10-class 2-group and 4-class 3-group it doesn’t improve worst-group and hurts overall accuracy; the learned weights collapse to the text group in every case. I’m planning to try changing the KL lambda (e.g. stronger so q doesn’t collapse as much) and also adding a spurious correlation by skewing class proportions per group, to see if we can get clearer positive results. When are you free to meet to discuss?

Thank you!

Best,  
Arjun
