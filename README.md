# Chest X-Ray Pneumonia Classifier

A deep learning model that detects pneumonia from chest X-ray images using Transfer Learning with ResNet18 and PyTorch.

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **89.58%** |
| Model | ResNet18 (pretrained on ImageNet) |
| Training Samples | 5,216 |

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

- **Normal:** Healthy chest X-rays
- **Pneumonia:** X-rays showing bacterial or viral pneumonia

## Approach

### Why Transfer Learning?
Training a CNN from scratch on medical images requires millions of samples. Instead, I used ResNet18 — a model pretrained on 1.2 million ImageNet images — and fine-tuned only the final classification layer for our 2-class problem (Normal vs Pneumonia). This allows the model to leverage rich visual features already learned from large-scale data.

### Architecture

### Training Details
- **Framework:** PyTorch
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 5
- **Augmentation:** Random horizontal flip, random rotation (±10°)

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision pillow matplotlib
```

### 2. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it as:

### 3. Train the model
```bash
python3 train.py
```

## Key Learnings
- Transfer learning dramatically reduces training time and data requirements for medical imaging tasks
- Data augmentation (flipping, rotation) helps prevent overfitting on limited medical datasets
- Even with just 5 epochs on CPU, ResNet18 achieves ~90% accuracy on this binary classification task

## Tech Stack
- Python 3.9
- PyTorch 2.8
- torchvision
- ResNet18
  
