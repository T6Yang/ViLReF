RESUME=/path/to/your/model/dir/

VISION_MODEL=ViT-B-16
# VISION_MODEL=RN50

LOGS=/path/to/your/logging/dir/

NAME=default

export MASTER_PORT=8530
export CUDA_VISIBLE_DEVICES=2
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="localhost"
python ViLReF/training/test.py --logs ${LOGS} --name ${NAME} --resume ${RESUME} --vision-model ${VISION_MODEL} --use_visual