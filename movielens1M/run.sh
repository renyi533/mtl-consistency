#!/bin/sh -x
set -e
set -u


echo "Start @ `date +'%F %T'`"

gpu_device=2

models_dir='./chkpt'
profiler_dir='./progiler'
train_data_dir='./data/train/task1'
test_data_dir='./data/test/task1'

if [ -d $models_dir ]; then
    rm -r $models_dir
fi

# train
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py $train_data_dir $models_dir $profiler_dir training 
# eval
time CUDA_VISIBLE_DEVICES=$gpu_device python ./model/train.py $test_data_dir $models_dir $profiler_dir eval

echo "Done @ `date +'%F %T'`"
