# 🩺 Pneumonia Detection from Chest X-rays using DenseNet121

This project leverages **deep learning** and **transfer learning** to classify **chest X-ray images** as either **Pneumonia** or **Normal**.  
It uses a pretrained **DenseNet121** model, fine-tuned on the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## 📂 Dataset

- **Source:** [Kaggle - Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Classes:** `PNEUMONIA`, `NORMAL`  
- **Structure:**
- **Preprocessing:**
- Resized to **224×224**
- Pixel normalization to `[0,1]`
- Data augmentation (rotation, zoom, flips)

---

## 🧠 Model Architecture

- **Base Model:** DenseNet121 (ImageNet weights)
- **Additional Layers:**
- Global Average Pooling
- Dense(128, activation='relu')
- Dense(1, activation='sigmoid')
- **Training strategy:**
1. Freeze base layers → Train top layers
2. Unfreeze last 50 layers → Fine-tune the top of the model

---

## ⚙️ Training Details

- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam (`1e-4` for initial training, `1e-5` for fine-tuning)
- **Metrics:** Accuracy
- **Callbacks:** EarlyStopping (patience=3, restore best weights)
- **Batch Size:** 32
- **Image Size:** 224×224

---

## 📊 Results

| Metric            | Value     |
|-------------------|-----------|
| Training Accuracy | XX.XX %   |
| Validation Acc.   | XX.XX %   |
| Test Accuracy     | YY.YY %   |

> Replace `XX.XX` and `YY.YY` with actual values after training.  
> The notebook prints the test accuracy as:  
> ```python
> print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
> ```

---

## 📈 Visualizations

![Accuracy Plot](images/accuracy_plot.png)  
*Training vs. Validation Accuracy (including fine-tuning phase)*

![Loss Plot](images/loss_plot.png)  
*Training vs. Validation Loss over epochs*

> Save your plots as `images/accuracy_plot.png` and `images/loss_plot.png` inside an `images` folder in your repo.

---

## 🚀 Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
