# Tumor Detection from Chest X-Ray via Multi-Label Deep Learning

A multi-label classification system for detecting tumor findings in chest X-ray images,
built with PyTorch and DenseNet169. Designed for high-recall medical imaging scenarios
where multiple pathologies may co-occur.

---

## Motivation

Medical imaging datasets often suffer from **class imbalance** and **multi-label targets**
(a single X-ray may show multiple findings simultaneously). This project investigates how
modern deep learning techniques — including class-weighted loss, per-class threshold
optimization, and Stochastic Weight Averaging — can address these challenges in a
clinically realistic setting.

---

## Methods & Technical Highlights

### Model Architecture
- **Backbone**: DenseNet169 (ImageNet pretrained, fine-tuned)
- **Classifier head**: Linear(1664→512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512→N)
- **Output**: Multi-label sigmoid predictions (one score per pathology class)

### Preprocessing Pipeline
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the L channel in LAB
  color space — improves visibility of low-contrast lesions in X-rays
- Gaussian blur for noise reduction
- Resize to 320×320
- ImageNet normalization

### Training Strategy
| Technique | Purpose |
|-----------|---------|
| `BCEWithLogitsLoss` with `pos_weight` | Handles severe class imbalance |
| `CosineAnnealingWarmRestarts` | Avoids local minima, improves generalization |
| Automatic Mixed Precision (AMP) | Speeds up training on GPU |
| Gradient clipping (`max_norm=1.0`) | Stabilizes training |
| Stochastic Weight Averaging (SWA) | Improves final model generalization |
| Early stopping (patience=8) | Prevents overfitting |
| Checkpoint saving | Resumes training from best F1 state |

### Per-Class Threshold Optimization
Rather than applying a fixed threshold (e.g., 0.5) across all classes, this system
performs **per-class threshold search** over [0.1, 0.9] on the validation set,
maximizing per-class F1. This is critical for imbalanced multi-label settings where
different pathologies have different base rates.

---

## Evaluation Metrics
- **F1 Macro** (primary metric, saved model criterion)
- Hamming Loss
- Cohen's Kappa
- Matthews Correlation Coefficient

---

## Project Structure

```
├── train.py          # Training pipeline with SWA and AMP
├── test.py           # Inference on held-out test set
├── Data/             # X-ray images
├── excel/            # Labels file (.xlsx) with train/test split
├── Documents/        # Additional documentation
├── model_Allison.pt  # Saved best model weights
└── results_Allison.xlsx  # Test set predictions
```

---

## Requirements

```bash
pip install torch torchvision opencv-python pandas scikit-learn tqdm openpyxl
```

---

## Usage

**Training:**
```bash
python train.py
```

**Inference:**
```bash
python test.py --path /path/to/project --split test
```

---

## Key Design Decisions & Observations

1. **Why CLAHE?** X-ray images have low global contrast. CLAHE enhances local contrast
   without over-amplifying noise, which empirically improved F1 in early experiments.

2. **Why per-class thresholds?** A single threshold of 0.5 severely underperforms on
   rare classes. Per-class tuning on validation data yielded meaningful F1 improvements
   on minority labels.

3. **Why SWA?** SWA averages weights across late training epochs, landing in flatter
   loss minima that generalize better — especially valuable with small medical datasets.

4. **Limitation**: The current CNN baseline (2-layer) is defined but DenseNet169 is used
   in practice. Future work could explore Vision Transformers (ViT) or EfficientNet as
   alternative backbones.

---

## Future Directions
- Evaluate model performance across patient subgroups (age, sex, image quality)
  to assess fairness and robustness — a critical consideration in clinical deployment
- Incorporate uncertainty quantification (e.g., Monte Carlo Dropout) to flag
  low-confidence predictions for radiologist review
- Explore attention-based interpretability (GradCAM) to localize lesion regions

---

## Acknowledgements
Dataset and project framework from GWU Machine Deep Learning coursework.
