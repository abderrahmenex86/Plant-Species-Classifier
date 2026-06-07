# Flora

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/TorchMetrics-000000?style=for-the-badge&logo=pytorch&logoColor=white" alt="TorchMetrics" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </p>
</div>

Flora is a deep learning training and evaluation pipeline built in PyTorch to classify plant species. It generates the high-efficiency classification weights used directly within the [FloraLens](https://github.com/abderrahmenex86/FloraLens) on-device ecosystem.

## Features

- **Efficient Fine-Tuning**: Built around a MobileNetV3-Large backbone, using customized classification layers with a $0.5$ dropout rate to prevent overfitting.
- **Differential Learning Rates**: Optimizes training dynamics by updating the pretrained feature extractor at a lower learning rate (`1e-5`) while training the custom classifier head at a higher rate (`1e-3`).
- **Data Augmentations**: Real-time training pipeline using random horizontal and vertical flips, color jittering, and normalization.
- **Comprehensive Evaluation**: Measures and logs Top-1 Accuracy, Top-5 Accuracy, and Macro F1-Score over ~400 distinct classes from the Pl@ntNet-300K dataset.
- **Dynamic Learning Rate Scaling**: Utilizes `ReduceLROnPlateau` targeting the validation Macro F1 score.

## Tech Stack

- **Machine Learning:** PyTorch, TorchVision
- **Metrics Tracking:** TorchMetrics (Multiclass Accuracy, Multiclass F1-Score)
- **Data Handling & Pipelines:** PyTorch ImageFolder, DataLoader with pin memory and prefetch optimizations
- **Execution Utilities:** tqdm progress indicators, custom train/evaluation logging helpers

## Getting Started

### Prerequisites
- Python (v3.10+)
- CUDA-compatible GPU (highly recommended for training)
- Pl@ntNet-300K dataset (or any custom folder-structured image dataset)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/abderrahmenex86/flora.git
cd flora
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Data Preparation:**
Place the training and validation images inside a structured directory:

```Text
dataset/plantnet_300K/images/
├── train/
└── val/
```
4. **Run Training:**

```bash
python train.py
```

## Related Projects
- [FloraLens](https://github.com/abderrahmenex86/FloraLens) — The offline host Android application
- [Pesti](https://github.com/abderrahmenex86/pesti) — Pest classification model
- [Segmenti](https://github.com/abderrahmenex86/segmenti) — Disease segmentation model
