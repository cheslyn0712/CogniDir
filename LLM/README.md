# Large Language Models for Cognitive Malicious Comment Generation (CogniDir)

## Overview

This directory contains Large Language Models (LLMs) used for generating diverse cognitive malicious comments in the CogniDir framework. The framework employs multiple complementary LLMs to synthesize category-controllable malicious comments that approximate real-world attack patterns.

## Required Models

The following LLM models are required for the framework:

1. **Gemma-2-2B-IT** (`gemma-2-2b-it/`)
   - Download: [Hugging Face - google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
   - Model size: ~4GB
   - Usage: Used for generating diverse adversarial comments

2. **Mistral-7B-Instruct** (`Mistral-7B-Instruct-v0___3/`)
   - Download: [Hugging Face - mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
   - Model size: ~14GB
   - Usage: Used for generating category-specific attack comments

3. **Qwen2.5-32B** (`Qwen2.5-32B/`)
   - Download: [Hugging Face - Qwen/Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)
   - Model size: ~64GB
   - Usage: Used for generating high-quality adversarial comments with better controllability

## Model Setup

### Download Instructions

Run the following commands from the **project root** (paths are relative to the project root):

```bash
pip install huggingface_hub

# Download Gemma-2-2B-IT
huggingface-cli download google/gemma-2-2b-it --local-dir LLM/gemma-2-2b-it

# Download Mistral-7B-Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir LLM/Mistral-7B-Instruct-v0___3

# Download Qwen2.5-32B
huggingface-cli download Qwen/Qwen2.5-32B --local-dir LLM/Qwen2.5-32B
```

Alternatively, you can download models manually from the Hugging Face website and place them in the corresponding directories.

### Directory Structure

Each model directory should contain:
- `config.json`: Model configuration file
- `tokenizer.json` or `tokenizer_config.json`: Tokenizer files
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `vocab.txt` or similar: Vocabulary files
- Other model-specific files as required

