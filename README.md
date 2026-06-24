# CogniDir: Combating Cognitive Malicious Comments via Adaptive Distributional Learning for Robust Fake News Detection

This repository contains the implementation of **CogniDir**, an adaptive distributional learning framework for robust fake news detection against cognitive malicious comments.

## Key Features

- **Adaptive Distributional Learning**: Dynamically adjusts malicious comment proportions during training using InfoDirichlet Resampling (IDR)
- **LLM-Generated Cognitive Attacks**: Synthesizes diverse cognitive malicious comments using three complementary LLMs (Gemma-2-2B-IT, Mistral-7B-Instruct, Qwen2.5-32B) following a "understand-then-generate" prompting strategy
- **Multi-Expert Architecture**: Employs cross-attention mechanisms to integrate news content and multiple adversarial rationales
- **Robustness Enhancement**: Achieves significant improvements in F1 scores across different attack groups

### Training Process

During training, the CogniDir model employs the InfoDirichlet Resampling (IDR) mechanism:

1. The model is evaluated on validation splits corresponding to different attack groups (base, medium, strong)
2. Group-wise vulnerability scores are computed based on validation performance
3. These scores are mapped to Dirichlet distribution parameters to determine sampling weights
4. Training data is resampled with adjusted proportions, increasing exposure to the most vulnerable groups
5. This process iteratively narrows robustness gaps across different attack groups

## Directory Structure

```
CogniDir/
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
│
├── BERT/                     # Pre-trained BERT model directory
│   └── README.md            # BERT model download instructions
│
├── Data/                     # Datasets (see Data/README.md)
│   ├── README.md            # Dataset documentation
│   ├── Generation.py        # Cognitive malicious comment generation script
│   ├── Seed/                 # Raw seed data files
│   │   ├── Rumoureval.json
│   │   ├── Weibo16.json
│   │   └── Weibo20.json
│   └── Processed/           # Processed datasets with cognitive malicious comments
│       ├── rumour/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── test.json
│       ├── Weibo16/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── test.json
│       └── Weibo20/
│           ├── train.json
│           ├── val.json
│           └── test.json
│
├── LLM/                      # Large Language Models (see LLM/README.md)
│   ├── README.md            # LLM models documentation
│   ├── gemma-2-2b-it/       # Gemma-2-2B-IT model
│   ├── Mistral-7B-Instruct-v0___3/  # Mistral-7B-Instruct model
│   └── Qwen2.5-32B/         # Qwen2.5-32B model
│
└── Src/                      # Source code
    ├── modules/              # Core modules
    │   ├── __init__.py
    │   ├── initialized_proportion.py  # Initial sampling proportion module
    │   ├── fake_news_detector.py      # Fake news detector model
    │   ├── infodirichlet_resampling.py  # IDR mechanism
    │   ├── dataset.py       # Dataset classes
    │   ├── layers.py         # Neural network layers
    │   └── model.py         # Legacy model (CogniDirModel)
    ├── scripts/              # Training and evaluation scripts
    │   ├── train.py         # Training script
    │   ├── test.py          # Test script
    │   ├── main.py          # Legacy main script
    │   └── grid_search.py   # Hyperparameter search script
    └── utils/                # Utility functions
        ├── __init__.py
        ├── dataloader.py    # Data loading and preprocessing
        └── utils.py         # Helper functions
```

### Directory Descriptions

- **`BERT/`**: Directory for pre-trained BERT model files. See `BERT/README.md` for download instructions. Download `bert-base-chinese` for Chinese datasets or `bert-base-uncased` for English datasets.

- **`Data/`**:
  - `Seed/`: Raw seed data files containing original news articles without cognitive malicious comments
  - `Processed/`: Processed datasets with cognitive malicious comments (rationales) generated using LLMs. Each dataset contains train/val/test splits (70/15/15).
  - `Generation.py`: Script for generating cognitive malicious comments from seed data using three LLM models following the "understand-then-generate" strategy

- **`LLM/`**: Large Language Models used for generating cognitive malicious comments. See `LLM/README.md` for download instructions. Required models:
  - `gemma-2-2b-it/`: Gemma-2-2B-IT model
  - `Mistral-7B-Instruct-v0___3/`: Mistral-7B-Instruct model
  - `Qwen2.5-32B/`: Qwen2.5-32B model

- **`Src/`**: Core source code
  - `modules/`: Core modules including initialized proportion, fake news detector, InfoDirichlet resampling, and dataset classes
  - `scripts/`: Training and evaluation scripts
  - `utils/`: Utility functions for data loading, metrics computation, etc.

## Setup

### 1. Download BERT Model

Download a pre-trained BERT model and place files in `BERT/` directory. See `BERT/README.md` for detailed instructions.

### 2. Download LLM Models

Download the three required LLM models and place them in `LLM/` directory. See `LLM/README.md` for download instructions.

### 3. Generate Cognitive Malicious Comments

Generate cognitive malicious comments from seed data (run from the `Data/` directory):

```bash
cd Data
python Generation.py
```

This will process all seed datasets and generate cognitive malicious comments using the three LLM models. The processed datasets will be saved in `Data/Processed/`.

## Quick Start

All paths in scripts and documentation are **relative paths**. Training and testing scripts automatically switch the working directory to the project root, so run them from anywhere using paths such as `Data/Processed/rumour` and `BERT`.

### Training

Run from the project root (all paths below are relative to the project root):

```bash
python Src/scripts/train.py \
    --data_name rumour \
    --root_path Data/Processed/rumour \
    --bert_path BERT \
    --epoch xx \
    --batchsize xx \
    --lr xx
```

### Testing

```bash
python Src/scripts/test.py \
    --model_path param_model/CogniDir_rumour/1/parameter_bert.pkl \
    --root_path Data/Processed/rumour \
    --bert_path BERT \
    --group_wise
```

## Experimental Results

The CogniDir framework demonstrates significant improvements in robustness across three benchmark datasets, achieving substantial F1 score improvements while maintaining high accuracy. The adaptive distributional learning mechanism effectively narrows performance gaps across different attack groups.

