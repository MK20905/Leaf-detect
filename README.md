# 🌿 Leaf and Age Detection Challenge – Gromo25

This repository contains my solution for the **Gromo25 Leaf and Age Detection Challenge**, a computer vision task focused on estimating the **number of leaves** and **age of plants** from image data. The challenge was hosted as part of the ACM Gromo25 competition.

---

## 🧠 Problem Statement

Given a dataset of plant images captured at different growth stages, the objective is to develop models that can:

1. **Estimate plant age (in days)** – a regression task  
2. **Count the number of leaves** – a regression task

Each image may vary in quality, lighting, and orientation, making this a real-world vision challenge.

---

## 📁 Dataset Overview

- **Training Images:** ~8,000 plant images in `.jpg` format
- **Annotations:** A CSV file with columns: `image_id`, `age`, `leaf_count`
- **Test Set:** Unlabeled images for submission
- Data was split into 80% training and 20% validation for local evaluation

---

## 🧰 Tools & Frameworks

- **Python 3.10**
- **PyTorch** – Model training
- **OpenCV / PIL** – Image preprocessing
- **NumPy / Pandas** – Data manipulation
- **Albumentations** – Image augmentation
- **Matplotlib / Seaborn** – Visualization
- **scikit-learn** – Metrics and cross-validation

---

## 🏗️ Model Architecture

### 🖼️ Backbone
- Vision Transformer (ViT-B/16) and ResNet variants were used for feature extraction.
- Pretrained weights were loaded from ImageNet.

### 🔀 Fusion Techniques
- Multitask model shared the backbone with two output heads:
  - One for age prediction (regression)
  - One for leaf count (regression)
- Late fusion methods were explored post-submission using ensembling and feature blending.

---

## 🧪 Training & Evaluation

- Loss Function: MSELoss (Mean Squared Error) for both heads
- Optimizer: AdamW with learning rate schedulers
- Evaluation Metrics:
  - **RMSE** for Age
  - **RMSE** for Leaf Count
