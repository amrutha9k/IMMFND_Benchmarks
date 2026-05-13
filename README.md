# IMMFND_Benchmarks

## Indian Multilingual Multimodal Fake News Detection Benchmarks
IMMFND_Benchmarks is a benchmark repository for multilingual fake news detection experiments conducted on the IMMFND dataset.

The repository contains:

- Text-only transformer baselines
- Multimodal transformer-based fusion models
- Benchmarking across multilingual transformer architectures
- Experimental evaluations using multiple dropout configurations

The repository focuses on multilingual misinformation detection using both textual and visual modalities.

---

# Supported Models

## Text-Only Models

The following multilingual transformer models are benchmarked:

- mBERT (`bert-base-multilingual-cased`)
- MuRIL (`google/muril-base-cased`)
- XLM-RoBERTa (`xlm-roberta-base`)
- IndicBERTv2 (`ai4bharat/IndicBERTv2-MLM-Sam-TLM`)

---

# Multimodal Models

## Text + ResNet50 Fusion

A multimodal fusion architecture combining:

- multilingual transformer text embeddings
- ResNet50 image embeddings

using feature-level fusion for fake news classification.

---

## Text + CLIP Fusion

A multimodal architecture combining:

- multilingual transformer text embeddings
- CLIP visual embeddings (`openai/clip-vit-large-patch14`)

for multimodal misinformation detection.

---

# Experimental Configurations

The multimodal experiments were conducted using multiple dropout configurations:

- 0.3
- 0.4
- 0.5

The current repository release contains:

- training scripts
- experimental outputs
- benchmark results

corresponding to the **0.4 dropout configuration**.

Extended ablation studies and best-performing configurations will be released after ongoing evaluations and conference submissions.

---

# Repository Structure
```text
IMMFND_Benchmarks/
│
├── src/
│   └── training/
│       ├── train_text_models.py
│       ├── train_multimodal_resnet.py
│       └── train_multimodal_clip.py
│
├── data/
│   └── Final_Dataset/
│       │
│       ├── train/
│       │   │
│       │   ├── Fake/
│       │   │   ├── Fake_text.xlsx
│       │   │   └── Fake_image/
│       │   └── Real/
│       │       ├── Real_text.xlsx
│       │       └── Real_image/
│       │
│       ├── validation/
│       │   │
│       │   ├── Fake/
│       │   │   ├── Fake_text.xlsx
│       │   │   └── Fake_image/
│       │   │
│       │   └── Real/
│       │       ├── Real_text.xlsx
│       │       └── Real_image/
│       │
│       └── test/
│           │
│           ├── Fake/
│           │   ├── Fake_text.xlsx
│           │   └── Fake_image/
│           │
│           └── Real/
│               ├── Real_text.xlsx
│               └── Real_image/
│
├── demo_dataset/
│   └── sample_data_link.txt
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Dataset Structure

The training scripts expect the dataset to follow the exact directory structure shown above.

Each split must contain:

- Fake claims
- Real claims
- Corresponding image folders

---

# Required Excel File Format

The Excel files must contain the following columns:

| Column Name | Description |
|---|---|
| `claim` | Textual claim |
| `Sr. No` | Unique sample identifier used to map images |

---

# Image Naming Convention

Images inside:

- `Fake_image/`
- `Real_image/`

must be named using the corresponding `Sr. No` value.

Example:

```text
article101.jpg
article101.png
article102.jpg
```

---

# Demo Dataset

The complete IMMFND dataset is currently not publicly released due to ongoing research and conference submission processes.

A small demo dataset is provided for reproducibility and framework demonstration purposes.

## Demo Dataset Link

```text
<https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view>
```

The demo dataset preserves the same directory structure as the complete dataset.

---

# Installation

## 1. Clone the Repository

```bash
git clone <https://github.com/amrutha9k/IMMFND_Benchmarks>
cd IMMFND_Benchmarks
```

---

## 2. Install Dependencies

```
pip install -r requirements.txt
```

---

# Running Experiments

## Text-Only Baselines

```bash
python src/training/train_text_models.py
```

### Outputs

- trained checkpoints
- `model_results_text.csv`

---

# Multimodal ResNet50 Fusion

```bash
python src/training/train_multimodal_resnet.py
```

### Outputs

- multimodal checkpoints
- `resnet_multimodal_results_dropoutval.csv`

---

# Multimodal CLIP Fusion

```bash
python src/training/train_multimodal_clip.py
```

### Outputs

- multimodal checkpoints
- `clip_multimodal_results_dropoutval.csv`

---

# Training Details

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch Size | 32 |
| Learning Rate | 1e-5 |
| Max Sequence Length | 128 |
| Early Stopping Patience | 5 |

---

# Text-Only Fine-Tuning Results

| Model | Accuracy (%) | F1-Macro (%) | Precision (%) | Recall (%) |
|---|---|---|---|---|
| mBERT | 78.58 | 78.49 | 78.51 | 78.47 |
| MuRIL | 80.35 | 80.33 | 80.34 | 80.44 |
| XLM-RoBERTa | 79.52 | 79.47 | 79.45 | 79.51 |
| IndicBERTv2 | 81.51 | 81.50 | 81.55 | 81.66 |

---

# Multimodal Fine-Tuning Results (Dropout = 0.4)

## Text + ResNet50 Fusion

| Model | Accuracy (%) | F1-Macro (%) | Precision (%) | Recall (%) |
|---|---|---|---|---|
| mBERT + ResNet50 | 88.73 | 88.64 | 88.93 | 88.51 |
| MuRIL + ResNet50 | 88.47 | 88.41 | 88.49 | 88.36 |
| XLM-RoBERTa + ResNet50 | 88.76 | 88.68 | 88.87 | 88.59 |
| IndicBERTv2 + ResNet50 | 89.19 | 89.14 | 89.21 | 89.09 |

## Text + CLIP Fusion

| Model | Accuracy (%) | F1-Macro (%) | Precision (%) | Recall (%) |
|---|---|---|---|---|
| mBERT + CLIP | 90.07 | 89.96 | 90.55 | 89.77 |
| MuRIL + CLIP | 89.61 | 89.53 | 89.80 | 89.40 |
| XLM-RoBERTa + CLIP | 90.15 | 90.05 | 90.59 | 89.86 |
| IndicBERTv2 + CLIP | 90.23 | 90.15 | 90.47 | 90.00 |

---

# Experimental Results

The benchmark experiments demonstrate strong multilingual fake news detection performance across both textual and multimodal settings.

The repository currently contains:

- text-only benchmark results
- multimodal fusion benchmark results
- dropout 0.4 experimental configurations

Additional ablation studies and best-performing configurations will be released later.

---

# Features

- Multilingual fake news detection
- Transformer-based text classification
- Multimodal fusion architectures
- CLIP-based image feature extraction
- ResNet50-based image feature extraction
- Early stopping
- Mixed precision training
- Checkpoint saving
- Automatic evaluation and CSV export

---

# Citation

If you use this repository in research, please cite:

```text
IMMFND_Benchmarks: Indian Multilingual Multimodal Fake News Detection Benchmarking using Transformer and Vision-Language Models.
```

---

# License

This repository is intended for academic and research purposes.
