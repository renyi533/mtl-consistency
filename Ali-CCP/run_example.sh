set -x
rm -rf ckpt

python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode training --keep_prob 0.7  --model mmoe --co_attention True --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode pred --model mmoe --co_attention True --global_experience True --epoch 1 --pred_file ./mmoe_dml_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe --mode training --keep_prob 0.7  --model mmoe --co_attention False --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe --mode pred --model mmoe --co_attention False --global_experience False --epoch 1 --pred_file ./mmoe_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_attn --mode training --keep_prob 0.7  --model mmoe --co_attention True --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_attn --mode pred --model mmoe --co_attention True --global_experience False --epoch 1 --pred_file ./mmoe_attn_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_global --mode training --keep_prob 0.7  --model mmoe --co_attention False --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_global --mode pred --model mmoe --co_attention False --global_experience True --epoch 1 --pred_file ./mmoe_global_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_dml --mode training --keep_prob 0.7  --model ple --co_attention True --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_dml --mode pred --model ple --co_attention True --global_experience True --epoch 1 --pred_file ./ple_dml_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/ple --mode training --keep_prob 0.7  --model ple --co_attention False --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple --mode pred --model ple --co_attention False --global_experience False --epoch 1 --pred_file ./ple_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_attn --mode training --keep_prob 0.7  --model ple --co_attention True --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_attn --mode pred --model ple --co_attention True --global_experience False --epoch 1 --pred_file ./ple_attn_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_global --mode training --keep_prob 0.7  --model ple --co_attention False --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_global --mode pred --model ple --co_attention False --global_experience True --epoch 1 --pred_file ./ple_global_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/aitm --mode training --keep_prob 0.7  --model aitm --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/aitm --mode pred --model aitm --epoch 1 --pred_file ./aitm_dml_result.txt
