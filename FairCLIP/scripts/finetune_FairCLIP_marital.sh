#!/bin/bash
DATASET_DIR=/home/arch/Codes/FairCLIP/FUNDUS_Dataset/FairVLMed
RESULT_DIR=.
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=maritalstatus # Options: race | gender | ethnicity | language | maritalstatus
SUMMARIZED_NOTE_FILE=gpt-4_summarized_notes.csv
LR=1e-5
BATCH_SIZE=32
LAMBDA=1e-5
BATCH_SIZE_FAIR=32
SK_BLUR=5e-2

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_FairCLIP.csv

python ./finetune_FairCLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/results/glaucoma_FairCLIP_${MODEL_ARCH}_${ATTRIBUTE_TYPE} \
		--lr ${LR} \
		--batch_size ${BATCH_SIZE} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--attribute ${ATTRIBUTE_TYPE} \
		--batchsize_fairloss ${BATCH_SIZE_FAIR} \
		--lambda_fairloss ${LAMBDA} \
		--sinkhorn_blur ${SK_BLUR} \
		--seed 5681 \
		--summarized_note_file ${SUMMARIZED_NOTE_FILE} 