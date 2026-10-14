# Replication Guide: xCoRe (EMNLP 2025)

This guide covers how to replicate the results from:

> **xCoRe: Cross-context Coreference Resolution**
> Martinelli, Gatti, Navigli — EMNLP 2025
> https://aclanthology.org/2025.emnlp-main.1737

---

## Environment Setup

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

Requirements: Python ≥3.8, PyTorch with CUDA, see `setup.py` for full dependency list.

---

## Data Preparation

All datasets should be placed under a `data/` directory at the repo root. Expected paths:

| Dataset | Path |
|---------|------|
| LitBank train | `data/LitBank/train-splitted.english.jsonlines` |
| LitBank dev | `data/LitBank/dev.english.jsonlines` |
| LitBank test | `data/LitBank/test.english.jsonlines` |
| ECB+ train | `data/cross/ecb/train_s.jsonl` |
| ECB+ dev | `data/cross/ecb/dev.jsonl` |
| ECB+ test | `data/cross/ecb/test.jsonl` |
| SciCo train | `data/cross/scico/train_nosingletons.jsonl` |
| SciCo dev | `data/cross/scico/validation_nosingletons.jsonl` |
| SciCo test | `data/cross/scico/test.jsonl` |
| BookCoref train | `data/long/book/bookcoref/train/silver_train_simple_mes_hierarchical_splitted_1350_rateo.jsonlines` |
| BookCoref test | `data/long/book/bookcoref/full_test_final.jsonl` |
| Animal Farm test | `data/long/animalfarm/test.jsonlines` |

All files use the jsonlines coreference format with `doc_key`, `sentences`, and `clusters` fields.

---

## Training

All training is run from the **repo root** with:

```bash
python -m xcore.train [overrides]
```

Hydra config defaults are in `conf/root.yaml`. Override `data`, `model`, and `train` groups as shown below.

### Table 4: Long-document — xCoRe on LitBank

```bash
python -m xcore.train \
  data=xlitbank \
  model/cross=xdeberta-large_10000steps
```

Expected checkpoint val metric: ~0.782 CoNLL-F1 on LitBank test.

### Table 3: Cross-document — xCoRe on ECB+

Train on ECB+ with the 80k-step schedule (longer training improves cross-doc performance):

```bash
python -m xcore.train \
  data=xecb \
  model/cross=xdeberta-large_80000steps
```

Expected: ~40.3 CoNLL-F1 on ECB+ test (predicted mentions).

### Table 3: Cross-document — xCoRe^LitBank on ECB+

First train on LitBank (as above), then fine-tune on ECB+:

```bash
# Step 1: train on LitBank
python -m xcore.train \
  data=xlitbank \
  model/cross=xdeberta-large_10000steps

# Step 2: fine-tune on ECB+ from the LitBank checkpoint
python -m xcore.train \
  data=xecb \
  model/cross=xdeberta-large_80000steps \
  evaluation.checkpoint=/path/to/litbank_checkpoint.ckpt
```

Expected: ~42.4 CoNLL-F1 on ECB+ test.

### Table 3: Cross-document — xCoRe on SciCo

```bash
python -m xcore.train \
  data=xscico \
  model/cross=xdeberta-large_80000steps
```

Expected: ~27.8 CoNLL-F1 on SciCo test (predicted mentions).

### Table 3: xCoRe^LitBank on SciCo

```bash
# Step 1: train on LitBank (same as above)
# Step 2: fine-tune on SciCo
python -m xcore.train \
  data=xscico \
  model/cross=xdeberta-large_80000steps \
  evaluation.checkpoint=/path/to/litbank_checkpoint.ckpt
```

Expected: ~30.5 CoNLL-F1 on SciCo test.

### Table 4: Long-document — xCoRe on BookCoref

```bash
python -m xcore.train \
  data=xbookcoref \
  model/cross=xdeberta-large_10000steps
```

Expected: ~63.0 CoNLL-F1 on BookCoref test.

---

## Evaluation

Evaluation requires a trained checkpoint. Run from the **repo root**:

```bash
python -m xcore.evaluate \
  data=<dataset> \
  evaluation.checkpoint=/path/to/checkpoint.ckpt \
  evaluation.device=cuda
```

### Examples

**LitBank test set:**
```bash
python -m xcore.evaluate \
  data=xlitbank \
  evaluation.checkpoint=/path/to/litbank_best.ckpt
```

**ECB+ test set:**
```bash
python -m xcore.evaluate \
  data=xecb \
  evaluation.checkpoint=/path/to/ecb_best.ckpt
```

**SciCo test set:**
```bash
python -m xcore.evaluate \
  data=xscico \
  evaluation.checkpoint=/path/to/scico_best.ckpt
```

**Animal Farm (zero-shot from LitBank model):**
```bash
python -m xcore.evaluate \
  data=xanimalfarm \
  evaluation.checkpoint=/path/to/litbank_best.ckpt
```

Expected: ~42.2 CoNLL-F1 (Table 4).

---

## Expected Results Summary

### Table 3: Cross-document Coreference

| Model | ECB+ CoNLL-F1 | SciCo CoNLL-F1 |
|-------|--------------|----------------|
| xCoRe | 40.3 | 27.8 |
| xCoRe^LitBank | 42.4 | 30.5 |

### Table 4: Long-document Coreference

| Model | LitBank | Animal Farm | BookCoref |
|-------|---------|-------------|-----------|
| xCoRe | 78.2 | 42.2 | 63.0 |

All results use predicted mentions, singletons excluded, CoNLL-F1 = average of MUC, B³, CEAFe.

---

## Key Hyperparameters

| Parameter | Long-doc (10k) | Cross-doc (80k) |
|-----------|---------------|-----------------|
| Optimizer | Adafactor | Adafactor |
| Learning rate | 3e-5 | 3e-5 |
| Warmup steps | 10% of total | 10% of total |
| Total steps | 10,000 | 80,000 |
| Grad clip | 1.0 | 1.0 |
| Grad accumulation | 4 | 4 |
| Precision | 16-bit | 16-bit |
| Encoder | DeBERTa-v3-large | DeBERTa-v3-large |
| Cluster repr. | transformer | transformer |
| Span repr. | concat_start_end | concat_start_end |

---

## Pre-trained Checkpoints

The published xCoRe checkpoint (LitBank) is available on HuggingFace:

```python
from xcore.models.xcore_model import xCoRe
model = xCoRe("sapienzanlp/xcore-litbank", device="cuda")
```
