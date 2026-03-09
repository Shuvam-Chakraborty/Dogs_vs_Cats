# 🐱🐶 Cat vs Dog Image Classification

Binary image classifier using **ResNet-18 Transfer Learning** with data augmentation, trained on 2000 images.

---

## 📂 Dataset

**Dog_vs_Cat_Dataset** — 1000 cat images + 1000 dog images organized in class-named folders.

| Split | Size |
|---|---|
| Training | 70% (1400 images) |
| Validation | 15% (300 images) |
| Testing | 15% (300 images) |

---

## 🛠️ Tools
`MATLAB` · `Deep Learning Toolbox`

---

## 🔄 What the Code Does

| Step | Description |
|---|---|
| Data Loading | ImageDatastore with folder-based labels |
| Augmentation | Random rotation (±20°), horizontal flip, scale (0.8–1.2×) |
| Model | ResNet-18 with replaced FC + classification layers |
| Training | SGD with momentum, 15 epochs, LR = 1e-4, early stopping |
| Evaluation | Accuracy, Precision, Recall, F1-Score, Specificity, AUC |
| Visualization | Confusion matrix, ROC curve, misclassified images, confidence distribution |

---

## 📊 Output Metrics

- Test Accuracy, Validation Accuracy
- Per-class Precision, Recall, F1-Score, Specificity
- ROC Curve with AUC score
- Class-wise accuracy breakdown

---

## 🚀 How to Run

1. Unzip `Dog_vs_Cat_Dataset.zip` — ensure folder structure is `Dog_vs_Cat_Dataset/cat/` and `Dog_vs_Cat_Dataset/dog/`
2. Open `Cat_Dog_Classification_Project.m` in MATLAB
3. Run the script — training starts automatically
4. Model saved to `advanced_cat_dog_model.mat`

---

## 📁 Project Structure

```
Cat_Dog_Classification_Project.m     ← Main MATLAB script
Dog_vs_Cat_Dataset.zip               ← Dataset (cat/ and dog/ subfolders)
advanced_cat_dog_model.mat           ← Saved model (auto-generated)
```
