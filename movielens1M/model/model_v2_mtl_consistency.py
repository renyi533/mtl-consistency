# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys
import os
import tensorflow.compat.v1 as tf
import config

tf.disable_v2_behavior()

def input_fn(filenames, batch_size, num_epochs=1, perform_shuffle=False):
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        label1, label2, feat, feat_title, feat_genres = tf.split(columns.values, num_or_size_splits=[1,1,7,1,1])
        label1 = tf.string_to_number(label1, tf.float32)
        label2 = tf.string_to_number(label2, tf.float32)
        feat = tf.string_to_number(feat, tf.int64)
        feat_title = tf.string_to_number(tf.string_split(feat_title, ',').values, tf.int64)
        feat_genres = tf.string_to_number(tf.string_split(feat_genres, ',').values, tf.int64)      
        size = tf.reshape(tf.size(feat), [1])
        feat_dnn = feat % config.feature_size_dnn
        feat_title = feat_title % config.feature_size_dnn
        feat_genres = feat_genres % config.feature_size_dnn
        return {"size": size, "feat": feat, 'feat_dnn': feat_dnn, 'feat_title': feat_title, 'feat_genres': feat_genres, 'label2': label2}, label1

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=config.num_threads)  # .prefetch(500000)    # multi-thread pre-process then prefetch
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.padded_batch(batch_size, ({'size':[1], 'feat':[-1], 'feat_title':[-1], 'feat_genres':[-1],  'feat_dnn':[-1],  'label2':[1]}, [1]))
    dataset = dataset.prefetch(1)  # one batch
    # return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels1 = iterator.get_next()
    # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels1

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

def mmoe(dnn_inputs, mode=None):
    all_experts = []
    for i in range(config.expert_num):
        expert_input = dnn_inputs
        for j, node_num in enumerate(config.expert_layers):
            expert_input = tf.layers.dense(expert_input, node_num, activation=tf.nn.relu)
        all_experts.append(expert_input)

    task_outputs = []
    task_logits_inputs = []
    expert_config = config.task_experts
    for i in range(config.task_num):
        task_tower = []
        task_experts = [all_experts[k] for k in expert_config[i]]
        gate = tf.layers.dense(dnn_inputs, len(task_experts), activation=tf.nn.softmax)
        #tf.Print(gate, [gate], message="gate: ", first_n=10, summarize=5)
        gate = tf.expand_dims(gate, axis=1)
        task_experts = tf.stack(task_experts, axis=1)
        task_input = tf.squeeze(tf.matmul(gate, task_experts), axis=1)
         
        for j, node_num in enumerate(config.task_layers):
            task_input = tf.layers.dense(task_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task_input = tf.nn.dropout(task_input, keep_prob=config.dropout)
        task_logits_inputs.append(task_input)
        
        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=False)
        task_outputs.append(task_out)

    task_outputs_consistency = task_consistency(task_logits_inputs)
    for i in range(config.task_num):
        task_outputs[i] += task_outputs_consistency[i]

    return task_outputs

def ple(dnn_inputs, mode=None):
    all_experts = []
    for i in range(config.expert_num):
        expert_input = dnn_inputs
        for j, node_num in enumerate(config.expert_layers):
            expert_input = tf.layers.dense(expert_input, node_num, activation=tf.nn.relu)
        all_experts.append(expert_input)

    task_outputs = []
    task_logits_inputs = []
    expert_config = config.task_experts_ple
    for i in range(config.task_num):
        task_tower = []
        task_experts = [all_experts[k] for k in expert_config[i]]
        gate = tf.layers.dense(dnn_inputs, len(task_experts), activation=tf.nn.softmax)
        #tf.Print(gate, [gate], message="gate: ", first_n=10, summarize=5)
        gate = tf.expand_dims(gate, axis=1)
        task_experts = tf.stack(task_experts, axis=1)
        task_input = tf.squeeze(tf.matmul(gate, task_experts), axis=1)
         
        for j, node_num in enumerate(config.task_layers):
            task_input = tf.layers.dense(task_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task_input = tf.nn.dropout(task_input, keep_prob=config.dropout)
        task_logits_inputs.append(task_input)
        
        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=False)
        task_outputs.append(task_out)

    task_outputs_consistency = task_consistency(task_logits_inputs)
    for i in range(config.task_num):
        task_outputs[i] += task_outputs_consistency[i]

    return task_outputs
    
def model_fn(features, labels, mode=None, params=None):
 
    learning_rate = config.learning_rate
    feat_dnn, feat_title, feat_genres, size, labels2 = features['feat_dnn'], features['feat_title'], features['feat_genres'], features['size'], features['label2']

    with tf.variable_scope("mmoe"):
        DEEP_V_title = tf.get_variable(name='deep_v_title', shape=[config.feature_size_title, config.embedding_size_dnn], initializer=tf.glorot_normal_initializer())
        DEEP_V_genres = tf.get_variable(name='deep_v_genres', shape=[config.feature_size_genres, config.embedding_size_dnn], initializer=tf.glorot_normal_initializer())
        DEEP_V = tf.get_variable(name='deep_v', shape=[config.feature_size_dnn, config.embedding_size_dnn], initializer=tf.glorot_normal_initializer())

        dnn_inputs = tf.nn.embedding_lookup(DEEP_V, feat_dnn) # [None, F, embedding_size]
        dnn_inputs = tf.reshape(dnn_inputs, [-1, config.feature_num * config.embedding_size_dnn])

        dnn_inputs_title = tf.nn.embedding_lookup(DEEP_V_title, feat_title) # [None, F, embedding_size]
        dnn_inputs_title = tf.reduce_sum(dnn_inputs_title, 1) 
        dnn_inputs_genres = tf.nn.embedding_lookup(DEEP_V_genres, feat_genres) # [None, F, embedding_size]
        #dnn_inputs_genres = tf.reduce_sum(dnn_inputs_genres, 1) 
        dnn_inputs_genres = tf.reshape(dnn_inputs_genres, [-1, 6 * config.embedding_size_dnn])

        dnn_inputs = tf.concat([dnn_inputs, dnn_inputs_title, dnn_inputs_genres], axis=-1)

        if params and params['base']:
            if params['base']== 'mmoe':
                task_outputs = mmoe(dnn_inputs, mode)
            elif params['base']== 'ple':
                task_outputs = ple(dnn_inputs, mode)

        y_d_task1, y_d_task2 = task_outputs[0], task_outputs[1]
        
    predictions={'task1': tf.sigmoid(y_d_task1),'task2': y_d_task2}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d_task1, labels=labels)) \
            +  tf.losses.mean_squared_error(labels=labels2, predictions=y_d_task2)

   
    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
                       "auc": tf.metrics.auc(labels, tf.sigmoid(y_d_task1)),
                       "mse": tf.metrics.mean_squared_error(labels2, y_d_task2)
                       }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if config.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif config.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif config.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif config.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:  # 'sgd'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

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


