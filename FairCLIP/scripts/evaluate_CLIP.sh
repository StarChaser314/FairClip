#!/bin/bash
DATASET_DIR=/home/arch/Codes/FairCLIP/FUNDUS_Dataset/FairVLMed
RESULT_DIR=.
MODEL_ARCH=vit-l14  # Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}.csv

python ./evaluate_CLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/results/evaluation/glaucoma_CLIP_${MODEL_ARCH} \
		--lr ${LR} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		# --pretrained_weights '/home/arch/Codes/FairCLIP/FairCLIP/results/glaucoma_CLIP_vit-b16_seed9077_auc0.7181/clip_ep009.pth'