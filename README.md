# 🫁 Pneumonia Detection from Chest X-Rays

> Automated pneumonia screening using DenseNet121 transfer learning with Grad-CAM explainability — built as a clinical decision support aid.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Pneumonia is one of the leading causes of death globally, particularly in children under five. Radiologist availability is limited in low-resource settings, making automated screening tools clinically valuable.

This project fine-tunes a **DenseNet121** backbone (pretrained on ImageNet) to classify chest X-ray images as **NORMAL** or **PNEUMONIA**. It includes:

- Two-phase transfer learning (frozen base → selective fine-tuning)
- Class imbalance handling via balanced class weights
- Comprehensive evaluation: AUC-ROC, sensitivity, specificity, confusion matrix
- **Grad-CAM** heatmaps for visual explainability

---

## 📂 Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

> The dataset is imbalanced (~3:1 pneumonia-to-normal). Class weights are used during training to correct for this.

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    │
DenseNet121 (ImageNet pretrained)
    │  Phase 1: All layers frozen
    │  Phase 2: Last 50 layers unfrozen
    │
GlobalAveragePooling2D
    │
BatchNormalization
    │
Dense(256, ReLU) → Dropout(0.4)
    │
Dense(128, ReLU) → Dropout(0.3)
    │
Dense(1, Sigmoid)  →  P(Pneumonia)
```

**Why DenseNet121?** The CheXNet paper (Rajpurkar et al., 2017) demonstrated radiologist-level pneumonia detection using DenseNet121 on NIH ChestX-ray14 — making it a principled, literature-backed choice for this task.

---

## ⚙️ Training Strategy

### Phase 1 — Frozen Base
- DenseNet121 weights frozen, only the custom head is trained
- Learning rate: `1e-4`
- Prevents overwriting pretrained ImageNet features before the head stabilises

### Phase 2 — Fine-Tuning
- Last 50 DenseNet layers unfrozen
- Learning rate: `1e-5` (10× lower — avoids catastrophic forgetting)
- Adapts deep features to the X-ray domain

### Callbacks
| Callback | Monitored | Purpose |
|---|---|---|
| EarlyStopping | `val_auc` | Stop before overfitting |
| ReduceLROnPlateau | `val_loss` | Decay LR on plateau |
| ModelCheckpoint | `val_auc` | Save best weights |

---

## 📊 Evaluation Metrics

Accuracy is insufficient for imbalanced medical datasets. The model is evaluated on:

- **Sensitivity (Recall)** — fraction of true pneumonia cases caught *(clinical priority)*
- **Specificity** — fraction of true normal cases correctly identified
- **AUC-ROC** — overall discrimination ability across all thresholds
- **Confusion Matrix** — inspect false negatives (missed pneumonia) vs false positives
- **Classification Report** — precision, recall, F1 per class

> In clinical screening, **sensitivity is prioritised** — a missed pneumonia case is more dangerous than a false alarm.

---

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the X-ray influenced the prediction. A well-behaved model should focus on the **lung parenchyma** — opacities and consolidations — not on peripheral artifacts.

```
Original X-Ray  →  Grad-CAM Heatmap  →  Overlay
```

Explainability is critical for medical AI — clinicians need to understand and trust model decisions before acting on them.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pneumonia-detection.git
```

### 2. Open in Google Colab
Upload `pneumonia_detection_improved.ipynb` to [Google Colab](https://colab.research.google.com/).

### 3. Set up Kaggle API
- Download your `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings)
- Run **Cell 1** and upload it when prompted

### 4. Run all cells in order
Each cell is labelled and self-contained. The full pipeline runs in under 30 minutes on a Colab T4 GPU.

---

## 📓 Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Kaggle setup & dataset download |
| 2 | EDA — class distribution, sample images |
| 3 | Configuration & class weight computation |
| 4 | Data augmentation & generators |
| 5 | DenseNet121 model architecture |
| 6 | Phase 1 training (frozen base) |
| 7 | Phase 2 fine-tuning |
| 8 | Training curves (accuracy, AUC, loss) |
| 9 | Evaluation — confusion matrix, ROC curve, report |
| 10 | Grad-CAM visualization |
| 11 | Save model |
| 12 | Single image inference (upload & predict) |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x, Keras |
| Model | DenseNet121 (ImageNet pretrained) |
| Explainability | Grad-CAM (custom implementation) |
| Evaluation | scikit-learn |
| Image Processing | OpenCV, Pillow |
| Visualisation | Matplotlib |
| Platform | Google Colab |
| Data Source | Kaggle API |

---

## 📈 Data Augmentation

Applied only to training data to simulate natural X-ray variability:

| Transform | Value | Reason |
|---|---|---|
| Rotation | ±10° | Patient positioning differences |
| Width/Height shift | 5% | Equipment framing variation |
| Zoom | 10% | Scale variation |
| Horizontal flip | Yes | Anatomically valid for X-rays |
| Vertical flip | **No** | Chest X-rays are anatomically oriented |

---

## 📚 References

- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
- Huang et al. (2017). *Densely Connected Convolutional Networks.* CVPR 2017.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV 2017.

---

## 🔭 Future Work

- Multi-label classification on NIH ChestX-ray14 or CheXpert (14 findings)
- Uncertainty estimation using Monte Carlo Dropout or deep ensembles
- Vision Transformer (ViT / Swin) backbone comparison
- External validation on independent hospital dataset
- Threshold optimisation for target sensitivity ≥ 0.95

---

## 📄 License

This project is licensed under the MIT License. Dataset usage is subject to [Kaggle's terms](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

<p align="center">Made for learning · open to contributions · built with TensorFlow</p>
