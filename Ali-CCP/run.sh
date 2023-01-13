#!/bin/sh -x
set -e
set -u


echo "Start @ `date +'%F %T'`"

gpu_device=2

# train
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode training --model mmoe --co_attention True --global_experience True --epoch 10 
# eval
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode eval --model mmoe --co_attention True --global_experience True --epoch 1  
# pred this also use sklearn to compute auc, auc in tf is not precise
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode pred --model mmoe --co_attention True --global_experience True --epoch 1
echo "Done @ `date +'%F %T'`"
