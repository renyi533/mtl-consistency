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
  for model in mmoe ple
  do
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_dml_$i --mode training --keep_prob 0.9,0.7,0.7  --model ${model} --co_attention True --global_experience True --epoch 10 > logs/${model}_dml_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_$i --mode training --keep_prob 0.9,0.7,0.7  --model ${model} --co_attention False --global_experience False --epoch 10 > logs/${model}_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_attn_$i --mode training --keep_prob 0.9,0.7,0.7  --model ${model} --co_attention True --global_experience False --epoch 10 > logs/${model}_attn_train_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_global_$i --mode training --keep_prob 0.9,0.7,0.7  --model ${model} --co_attention False --global_experience True --epoch 10 > logs/${model}_global_train_$i.log 2>&1 &
    wait_function
    
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_dml_$i --mode pred --model ${model} --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_$i --mode pred --model ${model} --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_attn_$i --mode pred --model ${model} --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --model_dir ./ckpt/ali-ccp/${model}_global_$i --mode pred --model ${model} --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_$i.txt > logs/${model}_global_test_$i.log 2>&1 & 
    wait_function
  done  
    
    python ./model/train.py --model_dir ./ckpt/ali-ccp/aitm_$i --mode training --keep_prob 0.9,0.7,0.7  --model aitm --epoch 10 > logs/aitm_train_$i.log 2>&1 &
    wait_function
    python ./model/train.py --model_dir ./ckpt/ali-ccp/aitm_$i --mode pred --model aitm --epoch 1 --pred_file logs/aitm_dml_result_$i.txt > logs/aitm_test_$i.log 2>&1 &
    wait_function

i=$[$i+1]
done

find logs -type f -name "*result*" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort
