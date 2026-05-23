# AdiVaani: Hindi-Marathi Neural Machine Translation

This repository contains the complete implementation for the AdiVaani Initiative hiring assessment (MISN Lab, IIT Delhi). The goal is to build low-resource NLP systems for Hindi-Marathi translation.

## Project Structure

```
├── data/              # Datasets and Tokenizer models
├── src/               # Source code
│   ├── data/          # Preprocessing and dataloading
│   ├── evaluation/    # BLEU, CHRF++ metrics and plotting
│   ├── models/        # Architectures: LSTM Seq2Seq, Custom BERT, GPT-2, Fusion
│   └── training/      # Trainers, checkpointing, and status logging
├── configs/           # YAML configuration files for experiments
├── checkpoints/       # Trained models, history JSONs, and test examples
├── report/            # Technical report and documentation
└── prepare_kaggle.py  # Utility to package the project for Kaggle execution
```

## Setup & Installation

**Prerequisites:**
- Python 3.10+
- PyTorch (compiled with CUDA support for GPU training)

```bash
# Install requirements
pip install torch transformers sentencepiece sacrebleu matplotlib pyyaml tqdm
```

## Running the Experiments

The project uses a joint Hindi-Marathi SentencePiece BPE tokenizer (32K vocabulary). 

### Part I: Classical NMT (LSTM Seq2Seq)

**1. Random Embeddings Baseline:**
```bash
python -m src.training.trainer_lstm --experiment lstm_random_hi2mr --epochs 30 --batch_size 64
```

**2. Pretrained BERT Embeddings (L3Cube):**
```bash
# Uses huggingface l3cube-pune/hindi-bert-v2 and marathi-bert-v2 for embeddings
python -m src.training.trainer_lstm_bert --experiment lstm_bert_hi2mr --epochs 30
```

### Part II: Transformer Pretraining and Fusion

**1. Pretrain BERT Encoder (Masked Language Modeling):**
```bash
python -m src.training.trainer_bert --experiment bert_pretrain --epochs 10 --batch_size 32
```

**2. Pretrain GPT-2 Decoder (Causal Language Modeling):**
```bash
python -m src.training.trainer_gpt2 --experiment gpt2_pretrain --epochs 10 --batch_size 32
```

**3. Train Fusion Model (Translation):**
```bash
python -m src.training.trainer_fusion --experiment fusion_hi2mr --epochs 10 --batch_size 32
```

## Evaluation

To evaluate any trained model and generate plots:
```bash
python -m src.evaluation.evaluate_final --experiment [experiment_name]
python -m src.evaluation.evaluate_fusion # For the Fusion model specifically
```

## Hardware & Environment
- **Local Testing:** Tested on NVIDIA RTX 4050 (6GB VRAM) with mixed precision (FP16).
- **Full Training:** Kaggle Notebooks (T4 x2) used for 30-epoch LSTM runs and Transformer pretraining due to compute constraints.

*See `report/TECHNICAL_REPORT.md` for a comprehensive discussion on architecture, optimization, and failure analysis.*
