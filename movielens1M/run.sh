#!/bin/sh -x
set -e
set -u


echo "Start @ `date +'%F %T'`"

gpu_device=2

models_dir='./chkpt'
# train_data_dir='./data/train'
# test_data_dir='./data/test'
train_data_dir='./data_shuffle/train'
test_data_dir='./data_shuffle/test'

if [ -d $models_dir ]; then
    rm -r $models_dir
fi

# train
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py $train_data_dir $models_dir mmoe training 
# eval
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py $test_data_dir $models_dir mmoe eval

echo "Done @ `date +'%F %T'`"
