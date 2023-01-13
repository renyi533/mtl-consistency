set -x
rm -rf ckpt

python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode training --model mmoe --co_attention True --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_dml --mode pred --model mmoe --co_attention True --global_experience True --epoch 1 --pred_file ./mmoe_dml_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe --mode training --model mmoe --co_attention False --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe --mode pred --model mmoe --co_attention False --global_experience False --epoch 1 --pred_file ./mmoe_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_attn --mode training --model mmoe --co_attention True --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_attn --mode pred --model mmoe --co_attention True --global_experience False --epoch 1 --pred_file ./mmoe_attn_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_global --mode training --model mmoe --co_attention False --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/mmoe_global --mode pred --model mmoe --co_attention False --global_experience True --epoch 1 --pred_file ./mmoe_global_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_dml --mode training --model ple --co_attention True --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_dml --mode pred --model ple --co_attention True --global_experience True --epoch 1 --pred_file ./ple_dml_result.txt

python ./model/train.py --model_dir ./ckpt/ali-ccp/ple --mode training --model ple --co_attention False --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple --mode pred --model ple --co_attention False --global_experience False --epoch 1 --pred_file ./ple_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_attn --mode training --model ple --co_attention True --global_experience False --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_attn --mode pred --model ple --co_attention True --global_experience False --epoch 1 --pred_file ./ple_attn_result.txt


python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_global --mode training --model ple --co_attention False --global_experience True --epoch 10
python ./model/train.py --model_dir ./ckpt/ali-ccp/ple_global --mode pred --model ple --co_attention False --global_experience True --epoch 1 --pred_file ./ple_global_result.txt
