# Caption Generation vs Classification for TextCaps

## 🎯 **The Question**

**TextCaps is designed for caption generation** (image → text), but we're currently doing **classification** (image → class label). Should we switch?

---

## 📊 **Current Approach: Classification**

### Architecture
```
Image → ResNet → z_v (64-dim)
Text → CharCNN → z_t (64-dim)
         ↓
    [z_v; z_t] → MLP Head → logits (10 classes)
```

### What We're Doing
- **Task**: Predict which of 10 classes an image belongs to
- **Loss**: Cross-entropy on class labels
- **Evaluation**: Accuracy, F1, Precision, Recall
- **Groups**: Visual (images) vs Text (captions) - both predict same class

### Pros ✅
- Simple to implement (fits existing codebase)
- Fast to train and evaluate
- Still demonstrates multi-modal alignment
- GroupDRO works directly (per-group classification loss)

### Cons ❌
- **Not the actual TextCaps task** (feels like a workaround)
- Less interesting for a research paper
- Doesn't test sequence generation capabilities
- Classification is easier than generation

---

## 🎨 **Alternative: Caption Generation**

### Architecture (Would Need)
```
Image → ResNet → z_v (64-dim)
         ↓
    Decoder (LSTM/Transformer)
         ↓
    Generated Caption (sequence of tokens)
```

### What We'd Be Doing
- **Task**: Generate text caption (both groups generate the same target caption)
- **Loss**: Sequence-to-sequence cross-entropy (per token)
- **Evaluation**: CIDEr, BLEU, METEOR, ROUGE
- **Groups** (still multi-modal!):
  - **Group 0: Visual → Caption** (image encoder → decoder → caption)
    - Input: Image
    - Encoder: ResNet (visual features)
    - Output: Generated caption
  - **Group 1: Text → Caption** (text encoder → decoder → caption)
    - Input: OCR text from image
    - Encoder: CharCNN/Transformer (text features)
    - Output: Generated caption (same target as Group 0)
  
**Why this still makes sense:**
- ✅ Tests heterogeneous feature spaces (visual vs text encoders)
- ✅ Both groups generate the same target caption
- ✅ GroupDRO balances performance: visual group might struggle with text-heavy images, text group might struggle with visual-heavy images
- ✅ Demonstrates robustness across different input modalities

### Pros ✅
- **The actual TextCaps task** (more authentic)
- More interesting for a conference paper
- Tests real multi-modal understanding
- Better demonstrates GroupDRO on heterogeneous tasks

### Cons ❌
- **Significant architecture changes needed**:
  - Replace classifier head with decoder (LSTM/Transformer)
  - Teacher forcing during training
  - Beam search/greedy decoding during inference
  - Sequence padding and masking
- **More complex training**:
  - Sequence losses (not just classification)
  - GroupDRO needs to work with variable-length sequences
  - Longer training time
- **Different evaluation**:
  - Need CIDEr/BLEU/METEOR/ROUGE metrics
  - More complex to implement

---

## 🔧 **What Would Need to Change**

### 1. **Model Architecture**
```python
# Current: Classification head
class MLPHead(nn.Module):
    def forward(self, z):
        return self.fc(z)  # → logits (num_classes)

# New: Decoder head
class CaptionDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, hidden_dim):
        self.lstm = nn.LSTM(latent_dim, hidden_dim, ...)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z, captions=None):
        # Teacher forcing during training
        # Autoregressive during inference
        return logits  # (seq_len, batch, vocab_size)
```

### 2. **Loss Function**
```python
# Current: Classification
loss = CrossEntropy(logits, class_labels)

# New: Sequence-to-sequence
loss = 0
for t in range(seq_len):
    loss += CrossEntropy(logits[t], target_tokens[t])
```

### 3. **Evaluation Metrics**
```python
# Current
accuracy = (pred == target).mean()
f1 = compute_f1(pred, target)

# New
cider_score = compute_cider(generated_captions, reference_captions)
bleu_score = compute_bleu(generated_captions, reference_captions)
meteor_score = compute_meteor(generated_captions, reference_captions)
rouge_score = compute_rouge(generated_captions, reference_captions)
```

### 4. **GroupDRO Adaptation**
- Current: Works with per-sample classification loss
- New: Need to aggregate sequence losses per group
- Could use average token-level loss per sample, then group-level aggregation

### 5. **Data Loading**
- Current: Each image → 2 samples (visual + text), both with same class label
- New: Each image → 2 samples:
  - **Visual group**: (image, target_caption_sequence)
    - Input: Image tensor
    - Target: Reference caption (tokenized)
  - **Text group**: (OCR_text, target_caption_sequence)
    - Input: OCR text from image (tokenized)
    - Target: Same reference caption (tokenized)
  
**Key insight**: Both groups generate the **same target caption**, but from different input modalities. This tests whether:
- Visual encoder can understand images well enough to generate captions
- Text encoder can understand OCR text well enough to generate captions
- GroupDRO can balance performance when one modality is harder than the other

---

## 🤔 **Recommendation**

### **Option A: Keep Classification** (Faster, Simpler)
- **Pros**: Already working, can get results quickly
- **Cons**: Not the "real" task, less impressive for paper
- **Best for**: Quick experiments, validating GroupDRO works on multi-modal data

### **Option B: Switch to Caption Generation** (More Authentic, Harder)
- **Pros**: Real TextCaps task, more impressive for paper
- **Cons**: Significant implementation work (1-2 weeks)
- **Best for**: Conference paper with more time

### **Option C: Hybrid Approach** (Best of Both?)
- Start with classification to validate GroupDRO works
- Then implement caption generation for final paper results
- Show both tasks in paper (classification as proof-of-concept, generation as main result)

---

## 💡 **My Take**

For a **conference paper**, I'd recommend **Option C**:

1. **Now**: Keep classification, get baseline + GroupDRO working well
2. **Next**: Implement caption generation (more work, but more impressive)
3. **Paper**: Show both:
   - Classification: "GroupDRO works on multi-modal classification"
   - Generation: "GroupDRO improves worst-group caption quality"

This gives you:
- ✅ Quick validation that GroupDRO works
- ✅ Impressive results for the paper
- ✅ Two different tasks showing robustness

---

## 🚀 **If We Switch to Caption Generation**

**Estimated work**: 1-2 weeks
- Decoder architecture: 2-3 days
- Sequence loss + training loop: 2-3 days
- Evaluation metrics (CIDEr, BLEU, etc.): 2-3 days
- GroupDRO adaptation: 1-2 days
- Testing & debugging: 2-3 days

**Would you like to**:
1. **Keep classification** for now (get results faster)?
2. **Switch to caption generation** (more authentic, more work)?
3. **Do both** (classification first, then generation)?
