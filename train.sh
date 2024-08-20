#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}
#export TMPDIR=/tmp/

export CUDA_VISIBLE_DEVICES=0
# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8540
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0
#export WORLD_SIZE=16

export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/

DATAPATH=../data

# data options
train_data=${DATAPATH}/datasets/tongren_v3_512/trainval_crossed_medclip/lmdb/train
val_data=${DATAPATH}/datasets/tongren_v3_512/trainval_crossed_medclip/lmdb/valid  # if val_data is not specif  ied, the validation will be automatically disabled
#train_data=${DATAPATH}/datasets/tongren_v3_512_N100000_H0.15/lmdb/train
#val_data=${DATAPATH}/datasets/tongren_v3_512_N100000_H0.15/lmdb/valid # if val_data is not specif  ied, the validation will be automatically disabled

# restore options
#resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231119_recurrent231113/checkpoints/epoch6.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231113_weightedlabels/checkpoints/epoch7.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231118_weightedlabels_ignoreneginf/checkpoints/epoch3.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231117_weightedlabels/checkpoints/epoch10.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231115_weightedlabels/checkpoints/epoch7.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231115_lockimgencoder/checkpoints/epoch7.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/alldata/231113_finetunevit/cliploss/lr3e-5/checkpoints/epoch3.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/weightedmax_v2_lr3e-5_tongren_v3_512-cn_finetune_vit-b-16_roberta-base_bs16_removeRotate/231124/cosineweight/pulloutother/checkpoints/epoch6.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/231011_tongren_v3_512-cn_v1_finetune_Deterministic_vit-b-16_roberta-base_bs16_removeRotate/checkpoints/epoch1.pt # or specify your customed ckpt path to resume
#resume=${DATAPATH}/experiments/231012_tongren_v3_512-cn_v3.1.5_finetune_Deterministic_vit-b-16_roberta-base_bs16_removeRotate/checkpoints/epoch1.pt
#resume=${DATAPATH}/experiments/231022_tongren_v3_512-cn_v1_finetune_Deterministic_vit-b-16_roberta-base_bs16_removeRotate/checkpoints/epoch6.pt
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
#reset_data_offset=""
#reset_optimizer=""
warmup=100

# output options
output_base_dir=${DATAPATH}/experiments/
name=ALBEF_ita_loss/240717_no_resume_ckpt/ViT
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=100
# batch_size=96
# valid_batch_size=96  # 不能设置为“1”，否则准确率为100%
batch_size=256
valid_batch_size=256  # 不能设置为“1”，否则准确率为100%
lr=1e-2
wd=0.001
max_epochs=20
valid_step_interval=999999
valid_epoch_interval=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment" # True
# use_augment=""

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} cn_clip/training/main.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --num-workers=0 \
          --resume=${resume} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --lr=${lr} \
          --wd=${wd} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          --text-model=${text_model} \
          --skip-aggregate \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --grad-checkpointing \
        #   --local_rank=${LOCAL_RANK} \
