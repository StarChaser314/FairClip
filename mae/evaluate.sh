DATA_DIR=/home/arch/Codes/FairCLIP/FUNDUS_Dataset/FairVLMed
FEATS_TYPE=multimodal # [image, multimodal]

PRETRAIN_CHKPT=ViT-L/14
EXP_NAME=evaluate_blip2_gpt-4
MODEL_TYPE=clip # [clip, blip2]
VISION_ENCODER_WEIGHTS=clip # [clip, pmc-clip]

# OMP_NUM_THREADS=1 python -m torch.distributed.launch \
#     --master_port=29501 \
#     --nproc_per_node=1 \
torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    main_linprobe.py \
    --model_type ${MODEL_TYPE} \
    --vl_feats_type ${FEATS_TYPE} \
    --blip_feats_select avgpool \
    --cfg-path ../LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml \
    --vision_encoder_weights ${VISION_ENCODER_WEIGHTS} \
    --summary_type gpt-4 \
    --batch_size 512 \
    --model vit_large_patch16 \
    --early_stopping \
    --patience 10 \
    --min_delta 0.0001 \
    --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 1000 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --data_path ${DATA_DIR} \
    --output_dir $EXP_NAME \
    --log_dir $EXP_NAME \
    --nb_classes 2 > ${EXP_NAME}.out