set -x
rm -rf ckpt

i="0"
iter="10"
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
    python ./model/train.py --val_data_dir ./data/video_games/val --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 100 > logs/${model}_dml_train_$i.log 2>&1 &
    python ./model/train.py --val_data_dir ./data/video_games/val --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_nostop_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 100 > logs/${model}_dml_nostop_train_$i.log 2>&1 &
    python ./model/train.py --val_data_dir ./data/video_games/val --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 100 > logs/${model}_train_$i.log 2>&1 &
    python ./model/train.py --val_data_dir ./data/video_games/val --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 100 > logs/${model}_attn_train_$i.log 2>&1 &
    python ./model/train.py --val_data_dir ./data/video_games/val --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_nostop_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 100 > logs/${model}_attn_nostop_train_$i.log 2>&1 &
    python ./model/train.py --val_data_dir ./data/video_games/val --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_global_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 100 > logs/${model}_global_train_$i.log 2>&1 &
    wait_function
    
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_dml_nostop_result_$i.txt > logs/${model}_dml_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_attn_nostop_result_$i.txt > logs/${model}_attn_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_global_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_$i.txt > logs/${model}_global_test_$i.log 2>&1 & 
    wait_function

    python ./model/train.py --test_data_dir ./data/video_games/train  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_train_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/train  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_dml_nostop_result_train_$i.txt > logs/${model}_dml_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/train  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_train_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/train  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_train_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/train  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_attn_nostop_result_train_$i.txt > logs/${model}_attn_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/train  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_global_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_train_$i.txt > logs/${model}_global_test_$i.log 2>&1 &
    wait_function

    python ./model/train.py --test_data_dir ./data/video_games/val  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --epoch 1 --pred_file logs/${model}_dml_result_val_$i.txt > logs/${model}_dml_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/val  --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_dml_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience True --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_dml_nostop_result_val_$i.txt > logs/${model}_dml_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/val --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience False --epoch 1 --pred_file logs/${model}_result_val_$i.txt > logs/${model}_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/val --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --epoch 1 --pred_file logs/${model}_attn_result_val_$i.txt > logs/${model}_attn_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/val --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_attn_nostop_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention True --global_experience False --co_attention_stop_grad False --epoch 1 --pred_file logs/${model}_attn_nostop_result_val_$i.txt > logs/${model}_attn_nostop_test_$i.log 2>&1 &
    python ./model/train.py --test_data_dir ./data/video_games/val --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/${model}_global_$i/best --mode pred --model ${model} --global_experience_attn_layer 1 --task_layers 128,80 --co_attention False --global_experience True --epoch 1 --pred_file logs/${model}_global_result_val_$i.txt > logs/${model}_global_test_$i.log 2>&1 &
    wait_function
  done  
    
    python ./model/train.py --train_data_dir ./data/video_games/train --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/aitm_$i --mode training --early_stop 5 --keep_prob 1.0,1.0,1.0  --model aitm --epoch 100 > logs/aitm_train_$i.log 2>&1 &
    wait_function
    python ./model/train.py --test_data_dir ./data/video_games/test --uniq_feature_cnt 24304,10672 --batch_size 512 --embedding_dim 24 --model_dir ./ckpt/video_games/aitm_$i/best --mode pred --model aitm --epoch 1 --pred_file logs/aitm_dml_result_$i.txt > logs/aitm_test_$i.log 2>&1 &
    wait_function

i=$[$i+1]
done

find logs -type f -name "*result*" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort

