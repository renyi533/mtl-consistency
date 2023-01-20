# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys
import os
import tensorflow.compat.v1 as tf
import config
from model import input_fn, model_fn
import argparse
from sklearn.metrics import roc_auc_score, mean_squared_error
from best_checkpoint_copier import BestCheckpointCopier

tf.disable_v2_behavior()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2intlist(v):
    return [int(i) for i in v.split(',')]

def str2floatlist(v):
    return [float(i) for i in v.split(',')]

def str2strlist(v):
    return [str(i) for i in v.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Run config.")
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Batch size.')
    parser.add_argument('--expert_num', type=int, default=3,
                        help='count of experts.')
    parser.add_argument('--task_num', type=int, default=2,
                        help='count of tasks.')
    parser.add_argument('--task_loss', type=str2strlist, default=['xent', 'xent'],
                        help='tasks\' loss. xent or mse')
    parser.add_argument('--task_weight', type=str2floatlist, default=[1.0, 1.0],
                        help='task weights to compute metric_sum for early stop, for mse should assign negative w')
    parser.add_argument('--uniq_feature_cnt', type=str2intlist, default=[238635,98,14,3,8,4,4,3,5,467298,6929,263942,106399,5888,104830,51878,37148,4],
                        help='feature cnt.')
    parser.add_argument('--expert_layers', type=str2intlist, default=[128],
                        help='expert layers.')
    parser.add_argument('--task_layers', type=str2intlist, default=[128,80],
                        help='task layers.')
    parser.add_argument('--embedding_dim', type=int, default=5,
                        help='Number of embedding dim.')
    parser.add_argument('--lamda', type=float, default=1e-6,
                        help='Regularizer weight.')
    parser.add_argument('--SNR_l0_loss_weight', type=float, default=0.001,
                        help='Regularizer weight.')
    parser.add_argument('--keep_prob', type=str2floatlist, default=[0.9,0.7,0.7],
                        help='Keep probability. 1: no dropout.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Specify an optimizer type (Adam, Adagrad, Sgd, Momentum).')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop')
    parser.add_argument('--co_attention', type=str2bool, default=True,
                        help='Whether to use co-attention')
    parser.add_argument('--global_experience', type=str2bool, default=True,
                        help='Whether to use global experience')
    parser.add_argument('--model', type=str, default='mmoe',
                        help='base model to test')
    parser.add_argument('--mode', type=str, default='training',
                        help='train or test')
    parser.add_argument('--model_dir', type=str, default='./ckpt',
                        help='model dir')
    parser.add_argument('--train_data_dir', type=str, default='./data/ali-ccp/train',
                        help='training data directory')
    parser.add_argument('--test_data_dir', type=str, default='./data/ali-ccp/test',
                        help='test data directory')
    parser.add_argument('--val_data_dir', type=str, default='./data/ali-ccp/val',
                        help='validation data directory')
    parser.add_argument('--pred_file', type=str, default='./pred.csv',
                        help='pred results')
    
    return parser.parse_args()



def get_feature_file(data_dir):
    infile_list = os.listdir(data_dir)
    infile_list = [data_dir + '/' + x for x in infile_list]
    return infile_list

def get_line_cnt(file_list):
    total_lines = 0
    for f in file_list:
        with open(f, 'r') as fp:
            num_lines = sum(1 for line in fp if line.rstrip())
            total_lines +=  num_lines
    return total_lines

def main(_):    
    args = parse_args()
    out_model_dir = args.model_dir 

    physical_devices = tf.config.list_physical_devices('GPU')
    print('Available gpu devices', physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    is_online_training, is_eval, is_pred = False, False, False
    if args.mode == 'training':
        is_online_training = True
    elif args.mode == 'eval':
        is_eval = True
    elif args.mode == 'pred':
        is_pred = True

    train_file_list = get_feature_file(args.train_data_dir)
    test_file_list = get_feature_file(args.test_data_dir)
    val_file_list = get_feature_file(args.val_data_dir)
    
    print(train_file_list, test_file_list, val_file_list) 
    train_line_cnt = get_line_cnt(train_file_list)
    val_line_cnt = get_line_cnt(val_file_list)
    test_line_cnt = get_line_cnt(test_file_list)
    steps_per_epoch = train_line_cnt // args.batch_size
    print('train_line_cnt:%d, val_line_cnt:%d, test_line_cnt:%d, steps_per_epoch:%d' %
          (train_line_cnt, val_line_cnt, test_line_cnt, steps_per_epoch))
    log_steps = 1000

    gpu_info = {}
    c = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count=gpu_info), log_step_count_steps=log_steps, save_summary_steps=log_steps, keep_checkpoint_max=5)

    if is_online_training:
      if args.early_stop <= 0: #disable early stop
        MTL = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params={'args': args}, config=c)
        MTL.train(input_fn=lambda: input_fn(train_file_list, batch_size=args.batch_size, num_epochs=args.epoch, perform_shuffle=True, task_num=args.task_num))
      else:
        MTL = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params={'args': args}, config=c)
        # steps = None
        max_steps = steps_per_epoch * args.epoch
        hook_list = [tf.train.ProfilerHook(save_steps=max(steps_per_epoch//10, log_steps), output_dir=out_model_dir, show_memory=True, show_dataflow=True),
                     tf.estimator.CheckpointSaverHook(save_steps=steps_per_epoch, checkpoint_dir=out_model_dir),
                     tf.estimator.experimental.stop_if_no_increase_hook(estimator=MTL, 
                        metric_name='metric_sum', max_steps_without_increase=steps_per_epoch * args.early_stop,
                        min_steps=steps_per_epoch, run_every_secs=None, run_every_steps=steps_per_epoch)
                     ]
        print(hook_list)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_file_list, batch_size=args.batch_size, num_epochs=args.epoch, perform_shuffle=True, task_num=args.task_num), max_steps=max_steps, hooks=hook_list)
        best_copier = BestCheckpointCopier(
            name='best', # directory within model directory to copy checkpoints to
            checkpoints_to_keep=2, # number of checkpoints to keep
            score_metric='metric_sum', # metric to use to determine "best"
            compare_fn=lambda x,y: x.score > y.score, # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
            sort_key_fn=lambda x: x.score,
            sort_reverse=True) # sort order when discarding excess checkpoints
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(val_file_list, batch_size=args.batch_size, num_epochs=1, task_num=args.task_num), steps=None, exporters=best_copier, start_delay_secs=120, throttle_secs=180)
        tf.estimator.train_and_evaluate(MTL, train_spec, eval_spec)          
    elif is_eval:
        MTL = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params={'args': args}, config=c)
        MTL.evaluate(input_fn=lambda: input_fn(test_file_list, batch_size=args.batch_size, num_epochs=1, task_num=args.task_num))
    elif is_pred:
        MTL = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params={'args': args}, config=c)
        tasks_p = []
        tasks_l = []
        for i in range(args.task_num):
            tasks_p.append([])
            tasks_l.append([])
        idx = 0
        with open(args.pred_file, 'w+') as f:
          for prob in MTL.predict(input_fn=lambda: input_fn(test_file_list, batch_size=args.batch_size, num_epochs=1, task_num=args.task_num)):
            for i in range(args.task_num):
                tasks_p[i].extend(prob['task%d'%i])
                tasks_l[i].extend(prob['task%d_l'%i])
            if idx % log_steps == 0:
                for i in range(args.task_num):
                    f.write('%f,%f;' %(prob['task%d'%i][0], prob['task%d_l'%i][0]))
                f.write('\n')
            idx = idx + 1
          for i in range(args.task_num):
              if args.task_loss[i] == 'xent':
                  metric = roc_auc_score(y_true=tasks_l[i], y_score=tasks_p[i])
                  f.write('task%d_auc:%f,' %(i, metric))
              else:
                  metric = mean_squared_error(y_true=tasks_l[i], y_pred=tasks_p[i])
                  f.write('task%d_mse:%f,' %(i, metric))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()



