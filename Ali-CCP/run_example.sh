set -x
rm -rf ckpt

i="0"
iter="5"
mkdir logs
wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}
while [ $i -lt $iter ]
do
  for model in mmoe ple snr
  do
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_dml_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 50 > logs/${model}_dml_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_dml_nostop_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 50 > logs/${model}_dml_nostop_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 50 > logs/${model}_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_attn_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 50 > logs/${model}_attn_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_attn_nostop_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 50 > logs/${model}_attn_nostop_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_global_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 50 > logs/${model}_global_train_$i.log 2>&1 &
    wait_function
    
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_dml_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_dml_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_dml_nostop_result_$i.txt > logs/${model}_dml_nostop_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_attn_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_attn_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_attn_nostop_result_$i.txt > logs/${model}_attn_nostop_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_global_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_$i.txt > logs/${model}_global_test_$i.log 2>&1 & 
    wait_function

    python ./model/train.py --test_data_dir ./data/ali-ccp/train  --model_dir ./ckpt/ali-ccp/${model}_dml_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_train_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/train  --model_dir ./ckpt/ali-ccp/${model}_dml_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_dml_nostop_result_train_$i.txt > logs/${model}_dml_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/train  --model_dir ./ckpt/ali-ccp/${model}_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_train_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/train  --model_dir ./ckpt/ali-ccp/${model}_attn_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_train_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/train  --model_dir ./ckpt/ali-ccp/${model}_attn_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_attn_nostop_result_train_$i.txt > logs/${model}_attn_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/train  --model_dir ./ckpt/ali-ccp/${model}_global_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_train_$i.txt > logs/${model}_global_test_$i.log 2>&1 &
    wait_function

    python ./model/train.py --test_data_dir ./data/ali-ccp/val  --model_dir ./ckpt/ali-ccp/${model}_dml_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_val_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/val  --model_dir ./ckpt/ali-ccp/${model}_dml_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_dml_nostop_result_val_$i.txt > logs/${model}_dml_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/val --model_dir ./ckpt/ali-ccp/${model}_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_val_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/val --model_dir ./ckpt/ali-ccp/${model}_attn_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_val_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/val --model_dir ./ckpt/ali-ccp/${model}_attn_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_attn_nostop_result_val_$i.txt > logs/${model}_attn_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/ali-ccp/val --model_dir ./ckpt/ali-ccp/${model}_global_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_val_$i.txt > logs/${model}_global_test_$i.log 2>&1 &
    wait_function
  done  
    
    python ./model/train.py --model_dir ./ckpt/ali-ccp/aitm_$i --mode training --early_stop 1 --keep_prob 1.0,1.0,1.0  --model aitm --epoch 50 > logs/aitm_train_$i.log 2>&1 &
    wait_function
    python ./model/train.py --model_dir ./ckpt/ali-ccp/aitm_$i/best --mode pred --model aitm --epoch 1 --pred_file logs/aitm_dml_result_$i.txt > logs/aitm_test_$i.log 2>&1 &
    wait_function

i=$[$i+1]
done

find logs -type f -name "*result*" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort
