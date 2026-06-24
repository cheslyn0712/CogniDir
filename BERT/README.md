# BERT Model Directory

## Overview

This directory should contain pre-trained BERT model files used for encoding news content and cognitive malicious comments in the CogniDir framework.

## Required Files

Download a pre-trained BERT model and place the following files in this directory:

- `config.json`: Model configuration file
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `tokenizer.json` or `tokenizer_config.json`: Tokenizer configuration
- `vocab.txt`: Vocabulary file

## Model Selection

Choose the appropriate BERT model based on your dataset language:

### For Chinese Datasets (Weibo16, Weibo20)

**Recommended**: `bert-base-chinese`

Download from Hugging Face (run from the project root):
```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download bert-base-chinese --local-dir BERT

# Or download manually from:
# https://huggingface.co/bert-base-chinese
```

### For English Datasets (rumour)

**Recommended**: `bert-base-uncased`

Download from Hugging Face (run from the project root):
```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download bert-base-uncased --local-dir BERT

# Or download manually from:
# https://huggingface.co/bert-base-uncased
```

## Directory Structure

After downloading, the `BERT/` directory should have the following structure:

```
BERT/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json (or tokenizer_config.json)
└── vocab.txt
```

## Usage

The model path is specified in training scripts via the `--bert_path` parameter, which defaults to `BERT` (relative to the project root). Ensure the model files are correctly placed before running training or evaluation.
