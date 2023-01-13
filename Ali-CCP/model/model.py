# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys
import os
import tensorflow.compat.v1 as tf
import config


def input_fn(filenames, batch_size, num_epochs=1, perform_shuffle=False):
    def decode_libsvm(line):
        columns = tf.string_split([line], ',')
        splits = [1] * 2
        splits.append(-1)
        label1, label2, feat = tf.split(columns.values, num_or_size_splits=splits)
        label1 = tf.string_to_number(label1, tf.float32)
        label2 = tf.string_to_number(label2, tf.float32)
        feat = tf.string_to_number(feat, tf.int64)
        size = tf.reshape(tf.size(feat), [1])
        return {"size": size, "feat": feat,  'label2': label2, 'label1': label1}, label1

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=config.num_threads)  # .prefetch(500000)    # multi-thread pre-process then prefetch
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size*10, reshuffle_each_iteration=True)
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.padded_batch(batch_size, ({'size':[1], 'feat':[-1],  'label2':[1], 'label1':[1]}, [1]))
    dataset = dataset.prefetch(1)  # one batch
    # return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels1 = iterator.get_next()
    # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels1

def task_coattention(task_tower_inputs):
    task_tower_inputs = tf.concat(task_tower_inputs, axis=-1)
    task_tower_inputs = tf.reshape(task_tower_inputs, [-1, config.task_num, config.expert_layers[-1]])

    task_tower_inputs = tf.layers.dense(task_tower_inputs, config.expert_layers[-1], activation=tf.nn.relu)

    weights = tf.matmul(task_tower_inputs, tf.transpose(task_tower_inputs, [0, 2, 1])) 
    weights /= task_tower_inputs.get_shape().as_list()[-1]** 0.5
    weights = tf.nn.softmax(weights)
    task_tower_inputs = tf.matmul(weights, task_tower_inputs)  
    task_tower_inputs = tf.reshape(task_tower_inputs, [-1, config.task_num * config.expert_layers[-1]])

    outputs=[]
    for i in range(config.task_num):
        output = task_tower_inputs[:, i * config.expert_layers[-1] : config.expert_layers[-1]*(i + 1)] 
        outputs.append(output)

    return outputs

def task_consistency(task_logits_inputs):
    task_outputs = []
    for i in range(config.task_num):
        task_input = task_logits_inputs[i]
        global_experience = tf.concat(task_logits_inputs, axis=-1)
        global_experience = tf.layers.dense(tf.stop_gradient(global_experience), config.task_layers[-1], activation=tf.nn.relu)
        experience_weight = tf.layers.dense(tf.concat([task_input, global_experience], axis=-1), config.task_layers[-1], activation=tf.nn.sigmoid)
        task_input = task_input + experience_weight * global_experience
        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=False)
        task_outputs.append(task_out)
    return task_outputs

def model(dnn_inputs, args, mode=None):
    if args.model == 'mmoe':
        task_tower_inputs = mmoe(dnn_inputs, args)
    elif args.model == 'ple':
        task_tower_inputs = ple(dnn_inputs, args)
    else:
        assert False
            
    if args.co_attention:
        task_tower_inputs = task_coattention(task_tower_inputs)
    task_outputs = []
    task_logits_inputs = []
    for i in range(config.task_num):
        task_input = task_tower_inputs[i]
        for j, node_num in enumerate(config.task_layers):
            task_input = tf.layers.dense(task_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task_input = tf.nn.dropout(task_input, keep_prob=args.keep_prob)
        task_logits_inputs.append(task_input)

        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=True)
        task_outputs.append(task_out)

    if args.global_experience:
        task_outputs_consistency = task_consistency(task_logits_inputs)
        for i in range(config.task_num):
            task_outputs[i] += task_outputs_consistency[i]
        
    return task_outputs

def mmoe(dnn_inputs, mode=None):
    all_experts = []
    for i in range(config.expert_num):
        expert_input = dnn_inputs
        for j, node_num in enumerate(config.expert_layers):
            expert_input = tf.layers.dense(expert_input, node_num, activation=tf.nn.relu)
        all_experts.append(expert_input)

    task_outputs = []
    task_tower_inputs = []
    expert_config = config.task_experts
    for i in range(config.task_num):
        task_tower = []
        task_experts = [all_experts[k] for k in expert_config[i]]
        gate = tf.layers.dense(dnn_inputs, len(task_experts), activation=tf.nn.softmax)
        #tf.Print(gate, [gate], message="gate: ", first_n=10, summarize=5)
        gate = tf.expand_dims(gate, axis=1)
        print(gate.get_shape())
        task_experts = tf.stack(task_experts, axis=1)
        print(task_experts.get_shape())
        task_input = tf.squeeze(tf.matmul(gate, task_experts), axis=1)
        task_tower_inputs.append(task_input)

    return task_tower_inputs

def ple(dnn_inputs, mode=None):
    all_experts = []
    for i in range(config.expert_num):
        expert_input = dnn_inputs
        for j, node_num in enumerate(config.expert_layers):
            expert_input = tf.layers.dense(expert_input, node_num, activation=tf.nn.relu)
        all_experts.append(expert_input)

    task_outputs = []
    task_tower_inputs = []
    expert_config = config.task_experts_ple
    for i in range(config.task_num):
        task_tower = []
        task_experts = [all_experts[k] for k in expert_config[i]]
        gate = tf.layers.dense(dnn_inputs, len(task_experts), activation=tf.nn.softmax)
        #tf.Print(gate, [gate], message="gate: ", first_n=10, summarize=5)
        gate = tf.expand_dims(gate, axis=1)
        task_experts = tf.stack(task_experts, axis=1)
        task_input = tf.squeeze(tf.matmul(gate, task_experts), axis=1)
        task_tower_inputs.append(task_input)

    return task_tower_inputs
    

def model_fn(features, labels, mode=None, params=None):
 
    args = params['args']
    feat_dnn, size, labels2 = features['feat'],  features['size'], features['label2']

    uniq_feature_cnt = args.uniq_feature_cnt
    slot_cnt = len(uniq_feature_cnt)
    dnn_inputs = []
    feat_dnn = tf.split(feat_dnn, num_or_size_splits=slot_cnt, axis=-1)
    with tf.variable_scope("model"):
        for i in range(slot_cnt):
            vocab_size = uniq_feature_cnt[i]
            DEEP_V = tf.get_variable(name='emb_%d' % i, 
                                     shape=[vocab_size, args.embedding_dim], 
                                     initializer=tf.glorot_normal_initializer())
            feature_emb = tf.nn.embedding_lookup(DEEP_V, feat_dnn[i])
            dnn_inputs.append(feature_emb)

        dnn_inputs = tf.concat(dnn_inputs, axis=-1) 
        dnn_inputs = tf.squeeze(dnn_inputs, axis=-2)
        
        task_outputs = model(dnn_inputs, args)
        y_d_task1, y_d_task2 = task_outputs[0], task_outputs[1]
    
    pred_task1 = tf.sigmoid(y_d_task1)
    pred_task2 = tf.sigmoid(y_d_task2)#*pred_task1    
    predictions={'task1': pred_task1,'task2': pred_task2,'task1_l': features['label1'], 'task2_l': labels2}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
    
    loss = tf.losses.log_loss(labels, pred_task1) + tf.losses.log_loss(labels2, pred_task2) 
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d_task1, labels=labels)) \
    #        +  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d_task2, labels=labels2))
   
    # Provide an estimator spec for `ModeKeys.EVAL`
    ROC_bucket = 20000
    eval_metric_ops = {
                       "auc1": tf.metrics.auc(labels, pred_task1, num_thresholds=ROC_bucket, curve='ROC'),
                       "auc2": tf.metrics.auc(labels2,pred_task2, num_thresholds=ROC_bucket, curve='ROC') 
                       }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if args.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif args.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=args.lr, initial_accumulator_value=1e-8)
    elif args.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.95)
    elif args.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(args.lr)
    else:  # 'sgd'
        optimizer = tf.train.GradientDescentOptimizer(args.lr)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    return train_op, loss, labels

if __name__ == '__main__':
    train_file_list = sys.argv[1]



