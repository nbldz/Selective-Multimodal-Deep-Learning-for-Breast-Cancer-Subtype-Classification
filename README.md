# Selective Multimodal Deep Learning for Breast Cancer Subtype Classification

Complete implementation of the research paper: "Selective Multimodal Deep Learning for Reliable Breast Cancer Subtype Classification from Histopathology and Genomic Data"

##  Overview

This repository provides an end-to-end pipeline for breast cancer subtype classification using multimodal deep learning. It integrates:
- **Histopathology images** (WSI patches) analyzed with CTransPath Vision Transformer
- **RNA-seq transcriptomic data** processed with deep neural networks
- **Smart routing mechanism** for optimal model selection based on prediction confidence
- **Attention rollout** for interpretable visualizations

##  Key Features

- **Multiple fusion strategies**: Concatenation, Gated Fusion, Cross-Attention
- **Uncertainty-aware routing**: Dynamically selects between RNA-only and multimodal models
- **Attention visualization**: Interprets which tissue regions influence predictions
- **95.05% accuracy** on TCGA-BRCA dataset
- **Clinical calibration**: Low Expected Calibration Error (ECE = 0.061)

##  Project Structure

```
.
├── main.py                      # Main training pipeline
├── inference.py                 # Inference script for new samples
├── attention_rollout.py         # Attention visualization module
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── setup_aws.sh                 # AWS setup script
├── README.md                    # This file
├── data/
│   ├── brca/                   # WSI patches organized by patient ID
│   │   ├── TCGA-E2-A10A/
│   │   │   ├── patch_001.jpg
│   │   │   ├── patch_002.jpg
│   │   │   └── ...
│   │   └── TCGA-*/
│   └── rna_seq.csv             # RNA-seq data with PAM50 labels
├── outputs/                     # Training outputs
│   ├── best_model.pth
│   ├── multimodal_*.pth
│   ├── cm_*.png
│   └── results_summary.json
└── visualizations/              # Attention maps and analysis
```

##  Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (tested on AWS A10G)
- 16GB+ RAM recommended

### Setup on AWS EC2

1. **SSH into your AWS instance:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

2. **Run setup script:**
```bash
chmod +x setup_aws.sh
./setup_aws.sh
```

3. **Activate environment:**
```bash
source venv/bin/activate
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

##  Data Preparation

### RNA-seq CSV Format

Your `rna_seq.csv` should have the following structure:

```csv
patient_id,gene1,gene2,gene3,...,geneN,PAM50
TCGA-E2-A10A,5.23,3.45,7.89,...,2.11,LumA
TCGA-A8-A08F,6.78,4.32,8.90,...,3.45,Her2
...
```

- **patient_id**: Patient identifier (must match folder names in WSI directory)
- **gene columns**: Gene expression values (raw counts or normalized)
- **PAM50**: Subtype label (one of: Basal, Her2, LumA, LumB, Normal)

### WSI Patches Structure

```
data/brca/
├── TCGA-E2-A10A/
│   ├── patch_001.jpg  (1000x1000 or 224x224)
│   ├── patch_002.jpg
│   └── ...
├── TCGA-A8-A08F/
│   ├── patch_001.jpg
│   └── ...
```

- Each patient has a folder with their ID
- Patches are JPG images (will be resized to 224x224)
- Recommended: 10-100 patches per patient

##  Usage

### 1. Configuration

Edit `config.yaml` or update paths in `main.py`:

```python
# In main.py, update Config class:
class Config:
    BRCA_WSI_DIR = "/path/to/brca/folder"
    RNA_CSV_PATH = "/path/to/rna_seq.csv"
    OUTPUT_DIR = "./outputs"
```

### 2. Training

Run the complete training pipeline:

```bash
# Basic training
python main.py

# Run in background with tmux (recommended for long training)
tmux new -s brca_training
python main.py
# Press Ctrl+B, then D to detach
# tmux attach -t brca_training  # to reattach
```

The pipeline will:
1. Preprocess RNA-seq and WSI data
2. Train RNA-only model
3. Train WSI-only model
4. Train multimodal models (3 fusion strategies)
5. Optimize routing threshold
6. Generate visualizations and reports

**Training time**: ~4-6 hours on A10G GPU for 1000 patients

### 3. Inference

#### Single Sample Prediction

```bash
python inference.py \
    --mode single \
    --model_dir ./outputs \
    --patient_id TCGA-E2-A10A \
    --rna_data ./data/sample_rna.json \
    --use_routing \
    --output ./prediction_result.json
```

#### Batch Prediction

```bash
python inference.py \
    --mode batch \
    --model_dir ./outputs \
    --batch_csv ./data/test_samples.csv \
    --use_routing \
    --output ./batch_predictions.csv
```

### 4. Attention Visualization

```python
from attention_rollout import visualize_predictions_with_attention
from main import Config

# Visualize attention for test samples
visualize_predictions_with_attention(
    model=trained_model,
    dataloader=test_loader,
    device=Config.DEVICE,
    num_samples=10,
    output_dir='./visualizations',
    class_names=Config.CLASS_NAMES
)
```

##  Results

Based on TCGA-BRCA dataset (1022 patients):

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| RNA-only | 91.00% | 0.90 | 0.91 |
| WSI-only | 62.00% | 0.57 | 0.64 |
| Multimodal (Concat) | 93.50% | 0.92 | 0.93 |
| Multimodal (Gated) | 94.20% | 0.93 | 0.94 |
| Multimodal (Cross-Attn) | 94.80% | 0.93 | 0.95 |
| **Routing-Based** | **95.05%** | **0.93** | **0.95** |

### Class-wise Performance (Routing Model)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Basal | 0.95 | 1.00 | 0.97 |
| Her2 | 0.88 | 0.88 | 0.88 |
| LumA | 0.98 | 0.94 | 0.96 |
| LumB | 0.90 | 0.95 | 0.93 |

### Calibration Metrics

- RNA-only ECE: 0.118
- Multimodal ECE: 0.094
- **Routing ECE: 0.061** ✓

##  Model Architecture

### RNA Encoder
- Input: 500 most variable genes
- Hidden layers: 512 → 512 neurons
- Activation: ReLU with Dropout (0.3)

### WSI Encoder (CTransPath)
- Vision Transformer (Swin-Tiny backbone)
- Input: 224×224 patches
- Output: 768-dimensional features
- Attention pooling for patch aggregation

### Fusion Strategies

1. **Concatenation**: Simple feature concatenation
2. **Gated Fusion**: Learnable gating mechanism
3. **Cross-Attention**: Transformer-style cross-modal attention (Best)

### Routing Mechanism

```
if RNA_confidence >= threshold (0.75):
    use RNA-only prediction
else:
    use Multimodal prediction
```

- Threshold optimized via Bayesian risk minimization
- Reduces computational cost while improving accuracy
- 38.2% of samples routed to multimodal model

##  Output Files

After training, `outputs/` contains:

```
outputs/
├── best_model.pth                          # RNA-only model
├── multimodal_concatenation.pth            # Concatenation fusion
├── multimodal_gated.pth                    # Gated fusion
├── multimodal_cross_attention.pth          # Cross-attention fusion
├── cm_rna_only.png                         # Confusion matrix
├── cm_multimodal_*.png                     # Confusion matrices
├── cm_routing.png                          # Routing model CM
├── classification_report_routing.png       # Performance heatmap
├── results_summary.json                    # All metrics
└── preprocessor.pkl                        # Fitted preprocessor
```

##  Visualizations

### Confusion Matrices
![Confusion Matrix Example](outputs/cm_routing.png)

### Attention Maps
Shows which tissue regions the model focuses on:
- High attention (red): Discriminative regions
- Low attention (blue): Less informative areas

### Attention Distribution Analysis
Analyzes attention patterns across different subtypes

##  Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
Config.BATCH_SIZE = 16  # or 8

# Reduce max patches
Config.MAX_PATCHES_PER_PATIENT = 30
```

### CUDA Not Available

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Missing Patches

The code handles missing patches gracefully, but ensure:
- Patient IDs in CSV match folder names exactly
- At least some patients have WSI patches
- Patches are valid JPG files

### Slow Training

```bash
# Use mixed precision (if supported)
# Add to training loop:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

##  Citation

If you use this code, please cite the original paper:

```bibtex
@article{hezil2025selective,
  title={Selective Multimodal Deep Learning for Reliable Breast Cancer Subtype Classification from Histopathology and Genomic Data},
  author={Hezil, Nabil and Bouridane, Ahmed and Hamoudi, Rifat and Al-maadeed, Somaya and Akbari, Younes and Abdullakutty, Faseela},
  journal={Medical Engineering \& Physics},
  year={2025}
}
```

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review configuration in `config.yaml`
3. Ensure data format matches requirements
4. Check GPU memory usage: `nvidia-smi`

## 📜 License

This implementation is for research purposes. Please refer to the original paper for licensing details.

##  Acknowledgments

- TCGA-BRCA dataset
- CTransPath pretrained model (Wang et al., 2022)
- PyTorch and timm libraries

---

**Happy Training! **