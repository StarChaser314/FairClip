# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FairCLIP is a research project that addresses fairness in vision-language learning, specifically in medical imaging. The project introduces the first fair vision-language medical dataset (Harvard-FairVLMed) and proposes FairCLIP, an optimal-transport-based approach to reduce bias in CLIP models across protected attributes (race, gender, ethnicity, language).

## Environment Setup

### Primary Environment
```bash
# Create conda environment from the provided environment file
conda env create -f fairclip.yml

# Or set up with uv (preferred for new setups)
uv sync
```

### PyTorch Version Update
The project originally used torch==1.11.0+cu113 but can be upgraded:
```bash
pip install torch===2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset Setup

The Harvard-FairVLMed dataset must be downloaded separately from the provided Google Drive link and placed in the expected directory structure:
```
FUNDUS_Dataset/FairVLMed/
├── Training/
├── Validation/
├── Test/
├── data_summary.csv
└── gpt-4_summarized_notes.csv
```

## Project Architecture

### Core Components

1. **FairCLIP Implementation** (`FairCLIP/`):
   - `finetune_FairCLIP.py`: Main training script for FairCLIP with fairness constraints
   - `finetune_CLIP.py`: Standard CLIP fine-tuning 
   - `evaluate_CLIP.py`: Evaluation script for CLIP models
   - `src/modules.py`: Core utilities including dataset classes, fairness metrics, and model components

2. **BLIP2 Integration** (`LAVIS/`):
   - Modified LAVIS framework for BLIP2 experiments
   - Configuration files in `lavis/projects/blip2/train/`

3. **Evaluation Scripts** (`src/`):
   - `blip_eval.py`: Comprehensive evaluation for BLIP/BLIP2 models
   - `clip_eval.py`: CLIP model evaluation
   - `fundus_dataloader.py`: Custom dataset loader for fundus images

### Key Data Structures

- **Protected Attributes**: Race (0: Asian, 1: Black, 2: White), Gender (0: Female, 1: Male), Ethnicity (0: Non-Hispanic, 1: Hispanic), Language (0: English, 1: Spanish, 2: Other)
- **Labels**: Binary classification for glaucoma detection (0: Non-Glaucoma, 1: Glaucoma)
- **NPZ Files**: Contain fundus images, clinical notes, demographic attributes, and labels

## Common Development Commands

### Training Commands

**FairCLIP Training:**
```bash
cd FairCLIP
python finetune_FairCLIP.py \
    --dataset_dir /path/to/FUNDUS_Dataset/FairVLMed \
    --result_dir ./results/glaucoma_FairCLIP_vit-b16_race \
    --lr 1e-5 \
    --batch_size 32 \
    --model_arch vit-b16 \
    --attribute race \
    --batchsize_fairloss 32 \
    --lambda_fairloss 1e-6 \
    --sinkhorn_blur 1e-4 \
    --summarized_note_file gpt4_summarized_notes.csv
```

**BLIP2 Pre-training:**
```bash
cd LAVIS
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 train.py \
    --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml
```

### Evaluation Commands

**CLIP/FairCLIP Evaluation:**
```bash
cd FairCLIP
python evaluate_CLIP.py \
    --dataset_dir /path/to/FUNDUS_Dataset/FairVLMed \
    --result_dir ./results/glaucoma_CLIP_vit-b16 \
    --model_arch vit-b16 \
    --pretrained_weights path-to-checkpoint/clip_ep002.pth
```

**BLIP2 Evaluation:**
```bash
python src/blip_eval.py \
    --cfg-path /path/to/config.yaml \
    --weights /path/to/checkpoint.pth \
    --vision_encoder_weights clip \
    --vl_type blip2 \
    --eval_type zero_shot \
    --prompt "A picture of " \
    --summary_type gpt-4
```

**Linear Probing:**
```bash
cd mae
python -m torch.distributed.launch --master_port=29501 --nproc_per_node=1 main_linprobe.py \
    --model_type blip2 \
    --vl_feats_type image \
    --cfg-path ../LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml \
    --vision_encoder_weights clip \
    --batch_size 512 \
    --epochs 1000 \
    --data_path /path/to/FUNDUS_Dataset/FairVLMed
```

### Data Processing

**LLM Summarization:**
```bash
python src/dataset_deidentification_summarization.py \
    --openai_key <YOUR_OPENAI_KEY> \
    --models gpt-4
```

## Key Configuration Parameters

- **Model Architectures**: `vit-b16`, `vit-l14` for CLIP models
- **Protected Attributes**: `race`, `gender`, `ethnicity`, `language`
- **Summary Types**: `original`, `gpt-4`, `pmc-llama`, `med42`
- **Evaluation Types**: `zero_shot`, `linear_probe`

## Fairness Evaluation Metrics

The project implements comprehensive fairness evaluation including:
- Demographic Parity Difference (DPD)
- Equalized Odds Difference (EOD)
- Equity-Scaled Accuracy (ES-ACC)
- Equity-Scaled AUC (ES-AUC)
- Between-group disparity metrics

## Important Notes

- All paths in scripts use placeholder values and need to be updated with actual dataset locations
- The project requires significant computational resources for training vision-language models
- Dataset access requires approval and adherence to the CC BY-NC-ND 4.0 license
- CUDA compatibility should be verified based on your system configuration