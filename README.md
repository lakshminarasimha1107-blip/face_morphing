# DeepGuard: CNN-Based Face Morphing Detection with 5-Fold Cross-Validation and Grad-CAM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning system for **Face Morphing and Swap Detection** using **ResNet-18 Transfer Learning**, **Stratified 5-Fold Cross-Validation** with leakage-free identity grouping, and **Grad-CAM Visual Explainability**.

---

## 1. Project Objective

The objective of this application is to distinguish authentic human face photographs (**REAL**) from manipulated face morphs/swaps (**MORPHED**). The system computes confidence probabilities using a calibrated softmax head and produces visual heatmaps using Gradient-weighted Class Activation Mapping (Grad-CAM) to pinpoint the facial regions that influenced the model's prediction.

---

## 2. Dataset Organization & Provenance

The system operates directly on the paired dataset located at:
`C:\Users\Narasimha\OneDrive\Documents\zip\zip`

```text
C:\Users\Narasimha\OneDrive\Documents\zip\zip\
├── real\              # 5,426 authentic face photographs (256x256 RGB JPEG)
├── fake\              # 5,426 morphed/swapped face photographs (256x256 RGB JPEG)
├── metadata.csv       # Pairwise identity mappings (target face and source face IDs)
└── README.md          # Dataset documentation from DeepGuard AI Project (2026)
```

### Dataset Inspection & Integrity Verification
- **Total Images:** 10,852 images (5,426 REAL + 5,426 FAKE)
- **Class Balance:** Perfectly balanced (50.0% REAL, 50.0% MORPHED)
- **Image Format:** All files are valid 256×256 3-channel RGB JPEGs (0 corrupted files)
- **Generation Method:** InsightFace `inswapper_128` (ONNX runtime)
- **Exact Duplicates:** 53 identical hash matches detected and automatically clustered to eliminate leakage.

---

## 3. Label Mapping

The project enforces consistent label representation across all modules:

```python
CLASS_TO_IDX = {
    "real": 0,
    "fake": 1
}

IDX_TO_CLASS = {
    0: "REAL",
    1: "MORPHED"
}
```

---

## 4. CNN Architecture: ResNet-18

- **Backbone:** ResNet-18 initialized with ImageNet-pretrained weights (`ResNet18_Weights.DEFAULT`).
- **Classification Head:** Linear layer with Dropout ($p=0.2$) mapping 512 feature maps to 2 output logits (`REAL` and `MORPHED`).
- **Input Dimensions:** 224 × 224 × 3 (RGB).
- **Optimization:** AdamW optimizer with initial learning rate $1 \times 10^{-4}$ and weight decay $1 \times 10^{-4}$.
- **Loss Function:** `CrossEntropyLoss`.
- **LR Scheduling:** `ReduceLROnPlateau` reducing learning rate on validation loss stagnation.
- **Overfitting Prevention:** Early stopping tracking validation loss with 5-epoch patience.

---

## 5. Stratified 5-Fold Cross-Validation (Zero Data Leakage)

To guarantee rigorous evaluation without optimistic bias:
1. **Paired Clustering:** Every `fake_XXXXX.jpg` is linked to `real_XXXXX.jpg`.
2. **Duplicate Clustering:** Exact hash duplicates are grouped.
3. **Identity Clustering:** Disjoint Set Union (Union-Find) groups source and target identities from `metadata.csv`.
4. **Stratified Group Splitting:** `StratifiedGroupKFold` assigns entire identity clusters to a single fold, ensuring no individual identity or duplicate appears in both train and validation splits.
5. **No Data Leakage:** Validation folds are strictly held out until epoch evaluation.

---

## 6. Evaluation Metrics

For every fold and aggregated across all five folds, the pipeline computes:
- **Accuracy**
- **Precision**
- **Recall / Sensitivity** (True Positive Rate for MORPHED faces)
- **Specificity** (True Negative Rate for REAL faces: $TN / (TN + FP)$)
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**

Aggregated results display the Mean ± Standard Deviation across all 5 folds.

---

## 7. Grad-CAM Visualization

Grad-CAM extracts feature activations and backpropagated gradients from the final convolutional layer of ResNet-18 (`model.backbone.layer4[-1]`):
1. Calculates $\alpha_k = \frac{1}{Z} \sum_{i}\sum_{j} \frac{\partial y_c}{\partial A^k_{i,j}}$.
2. Weights activation maps: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$.
3. Normalizes heatmap to $[0, 1]$ and resizes to original facial dimensions.
4. Overlays a colormap (Jet) onto the face to highlight suspicious manipulation regions.

*Disclaimer:* Grad-CAM highlights feature activations that contributed to the model's prediction. It does not provide pixel-level forensic boundaries.

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA drivers (optional, CPU mode is also supported)

### Step 1: Create and Activate Virtual Environment (PowerShell)
```powershell
cd C:\Users\Narasimha\OneDrive\Documents\zip\zip\face_morphing_detection
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

*(Note: For GPU acceleration with CUDA 12.1+, PyTorch can be installed via:*
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`*)*

### Step 3: Run the Streamlit Dashboard
```powershell
streamlit run app.py
```

---

## 9. Training via CLI (Optional)

You can also launch 5-fold training directly from the command line:

```powershell
python -m training.cross_validation
```

---

## 10. Project Directory Layout

```text
face_morphing_detection/
│
├── app.py                     # Streamlit multi-page dashboard
├── config.py                  # Centralized system configurations and paths
├── requirements.txt           # Dependency specifications
├── README.md                  # Comprehensive project documentation
│
├── data/
│   └── dataset_summary.json   # Generated dataset inspection summary
│
├── models/
│   ├── model.py               # ResNet-18 architecture with Grad-CAM hooks
│   └── checkpoints/           # Saved model checkpoints for each fold
│
├── training/
│   ├── dataset_inspector.py   # Dataset verification and README parsing
│   ├── dataset_loader.py      # PyTorch Dataset and leakage-free CV splitter
│   ├── train.py               # Training epoch loops and early stopping
│   ├── cross_validation.py    # 5-Fold cross-validation orchestrator
│   └── evaluate.py            # Checkpoint evaluation and summary tables
│
├── inference/
│   ├── predict.py             # Inference pipeline with confidence scores
│   └── gradcam.py             # Grad-CAM heatmap generator and overlay
│
├── utils/
│   ├── preprocessing.py       # EXIF transpose, augmentations, normalization
│   ├── metrics.py             # Complete classification metrics and aggregation
│   └── visualization.py       # ROC curves, confusion matrices, learning plots
│
├── results/
│   ├── fold_metrics.csv       # Per-fold validation metrics
│   ├── training_history.csv   # Per-epoch loss and accuracy histories
│   └── plots/                 # Saved high-resolution evaluation figures
│
└── prediction_history/        # Local prediction audit log (privacy controlled)
```

---

## 11. Troubleshooting

- **CUDA Out of Memory:** Reduce `batch_size` from 32 to 16 in `config.py` or via the training dashboard slider.
- **Dataset Path Changed:** Update `DATASET_ROOT` in `config.py`.
- **Training Interrupted:** Remove the lock file `results/.training_lock` if a previous run was abruptly terminated.
- **Port Conflict in Streamlit:** Launch with custom port: `streamlit run app.py --server.port 8502`.
