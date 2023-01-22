# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys
import os
import tensorflow.compat.v1 as tf
import config


def input_fn(filenames, batch_size, num_epochs=1, perform_shuffle=False, task_num=2):
    def decode_libsvm(line):
        columns = tf.string_split([line], ',')
        splits = [1] * task_num
        splits.append(-1)
        inputs = tf.split(columns.values, num_or_size_splits=splits)
        labels = []
        for i in range(task_num):
            labels.append(tf.string_to_number(inputs[i], tf.float32))
        feat = tf.string_to_number(inputs[task_num], tf.int64)
        size = tf.reshape(tf.size(feat), [1])
        ret_dict = {"size": size, "feat": feat}
        for i in range(task_num):
            ret_dict['label%d'%(i)] = labels[i]
        return ret_dict, labels[0]

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=config.num_threads)  # .prefetch(500000)    # multi-thread pre-process then prefetch
    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size*10, reshuffle_each_iteration=True)
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.padded_batch(batch_size, ({'size':[1], 'feat':[-1],  'label2':[1], 'label1':[1]}, [1]))
    dataset = dataset.prefetch(1)  # one batch
    # return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels1 = iterator.get_next()
    # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels1

def task_coattention(original_tower_inputs, args, mode):
    task_tower_inputs = tf.concat(original_tower_inputs, axis=-1)
    task_tower_inputs = tf.reshape(task_tower_inputs, [-1, args.task_num, args.expert_layers[-1]])

    task_tower_inputs_Q = tf.layers.dense(task_tower_inputs, args.expert_layers[-1], activation=tf.nn.relu)
    if args.co_attention_stop_grad:
        task_tower_inputs = tf.stop_gradient(task_tower_inputs)
    task_tower_inputs_K = tf.layers.dense(task_tower_inputs, args.expert_layers[-1], activation=tf.nn.relu)
    task_tower_inputs_V = tf.layers.dense(task_tower_inputs, args.expert_layers[-1], activation=tf.nn.relu)

    weights = tf.matmul(task_tower_inputs_Q, tf.transpose(task_tower_inputs_K, [0, 2, 1])) 
    weights /= task_tower_inputs.get_shape().as_list()[-1]** 0.5
    weights = tf.nn.softmax(weights)
    task_tower_inputs = tf.matmul(weights, task_tower_inputs_V)  
    task_tower_inputs = tf.reshape(task_tower_inputs, [-1, args.task_num * args.expert_layers[-1]])
    if mode == tf.estimator.ModeKeys.TRAIN:
        task_tower_inputs = tf.nn.dropout(task_tower_inputs, keep_prob=args.keep_prob[2])
        
    outputs=[]
    for i in range(args.task_num):
        output = task_tower_inputs[:, i * args.expert_layers[-1] : args.expert_layers[-1]*(i + 1)] + original_tower_inputs[i]
        outputs.append(output)

    return outputs

def task_consistency(task_logits_inputs, args, mode):
    task_outputs = []
    global_experience = tf.stop_gradient(tf.concat(task_logits_inputs, axis=-1))
    for i in range(args.task_num):
        task_input = task_logits_inputs[i]
        info_from_global = tf.layers.dense(global_experience, args.task_layers[-1], activation=tf.nn.relu)
        if mode == tf.estimator.ModeKeys.TRAIN:
            info_from_global = tf.nn.dropout(info_from_global, keep_prob=args.keep_prob[2])
        
        layer_input = tf.concat([task_input, info_from_global], axis=-1)
        for j in range(args.global_experience_attn_layer):
            if j == args.global_experience_attn_layer - 1:
                act_func = tf.nn.sigmoid
            else:
                act_func = tf.nn.relu
            layer_input = tf.layers.dense(layer_input, args.task_layers[-1], activation=act_func)
        experience_weight = layer_input
        task_input = task_input + experience_weight * info_from_global
        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=False)
        task_outputs.append(task_out)
    return task_outputs

def aitm(feature_embedding, args):
    l2_reg = tf.keras.regularizers.l2(args.lamda)
    all_weights = {}
    # attention
    all_weights['attention_w1'] = tf.get_variable(
        initializer=tf.random_normal(
            shape=[32, 32],
            mean=0.0,
            stddev=0.01),
        regularizer=l2_reg, name='attention_w1')  # k * k
    all_weights['attention_w2'] = tf.get_variable(
        initializer=tf.random_normal(
            shape=[32, 32],
            mean=0.0,
            stddev=0.01),
        regularizer=l2_reg, name='attention_w2')  # k * k
    all_weights['attention_w3'] = tf.get_variable(
        initializer=tf.random_normal(
            shape=[32, 32],
            mean=0.0,
            stddev=0.01),
        regularizer=l2_reg, name='attention_w3')  # k * k   
    
    def _attention( input1, input2):
        '''
        The attention module.
        :param input1: None, K
        :param input2: None, K
        :return: None, K
        '''
        # (N,L,K)
        inputs = tf.concat([input1[:, None, :], input2[:, None, :]], axis=1)
        # (N,L,K)*(K,K)->(N,L,K), L=2, K=32 in this.
        Q = tf.tensordot(inputs, all_weights['attention_w1'], axes=1)
        K = tf.tensordot(inputs, all_weights['attention_w2'], axes=1)
        V = tf.tensordot(inputs, all_weights['attention_w3'], axes=1)
        # (N,L)
        a = tf.reduce_sum(tf.multiply(Q, K), axis=-1) / \
            tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
        a = tf.nn.softmax(a, axis=1)
        # (N,L,K)
        outputs = tf.multiply(a[:, :, None], V)
        return tf.reduce_sum(outputs, axis=1)  # (N, K)
    
    def _gen_tower(feature_embedding, info, idx):
      with tf.variable_scope('task_%d' % idx):
        tower_out = tf.keras.layers.Dense(
            128, activation='relu')(feature_embedding)
        tower_out = tf.keras.layers.Dropout(
            1 - args.keep_prob[0])(tower_out)
        tower_out = tf.keras.layers.Dense(
            64, activation='relu')(tower_out)
        tower_out = tf.keras.layers.Dropout(
            1 - args.keep_prob[1])(tower_out)
        tower_out = tf.keras.layers.Dense(
            32, activation='relu')(tower_out)
        tower_out = tf.keras.layers.Dropout(
            1 - args.keep_prob[2])(tower_out)
        if info is None:
            ait = tower_out
        else:
            ait = _attention(tower_out, info)

        info = tf.keras.layers.Dense(
            32, activation='relu')(
            ait)
        info = tf.keras.layers.Dropout(
            1 - args.keep_prob[2])(info)
        return ait, info
    
    task_outputs = []
    info = None
    for i in range(args.task_num):
        ait, info = _gen_tower(feature_embedding, info, i)
        task_out = tf.keras.layers.Dense(1)(ait)   
        task_outputs.append(task_out)
    return task_outputs
     
def model(dnn_inputs, args, mode):
    snr_l0_loss = 0.0
    print('start to train model:%s' % args.model)
    if args.model == 'mmoe':
        task_tower_inputs = mmoe(dnn_inputs, args, mode)
    elif args.model == 'ple':
        task_tower_inputs = ple(dnn_inputs, args, mode)
    elif args.model == 'mssm':
        task_tower_inputs, snr_l0_loss = mssm(dnn_inputs, args, mode, args.mssm_level_cnt)
    elif args.model == 'snr_v2':
        task_tower_inputs, snr_l0_loss = mssm(dnn_inputs, args, mode, args.snr_level_cnt, False, False)
    elif args.model == 'snr':
        task_tower_inputs, snr_l0_loss = snr_v1(dnn_inputs, args, mode)
    elif args.model == 'se':
        task_tower_inputs = shared_embedding(dnn_inputs, args, mode)
    elif args.model == 'sb':
        task_tower_inputs = shared_bottom(dnn_inputs, args, mode)
    elif args.model == 'aitm':
        return aitm(dnn_inputs, args), snr_l0_loss
    else:
        assert False
            
    if args.co_attention:
        task_tower_inputs = task_coattention(task_tower_inputs, args, mode)
    task_outputs = []
    task_logits_inputs = []
    for i in range(args.task_num):
        task_input = task_tower_inputs[i]
        for j, node_num in enumerate(args.task_layers):
            task_input = tf.layers.dense(task_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task_input = tf.nn.dropout(task_input, keep_prob=args.keep_prob[1])
        task_logits_inputs.append(task_input)

        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=True)
        task_outputs.append(task_out)

    if args.global_experience:
        task_outputs_consistency = task_consistency(task_logits_inputs, args, mode)
        for i in range(args.task_num):
            task_outputs[i] += task_outputs_consistency[i]
        
    return task_outputs, snr_l0_loss

def mmoe(dnn_inputs, args, mode):
    all_experts = expert_layer([dnn_inputs] * args.expert_num, args, mode)

    task_tower_inputs = []
    for i in range(args.task_num):
        task_tower = []
        task_experts = [all_experts[k] for k in range(args.expert_num)]
        gate = tf.layers.dense(dnn_inputs, len(task_experts), activation=tf.nn.softmax)
        #tf.Print(gate, [gate], message="gate: ", first_n=10, summarize=5)
        gate = tf.expand_dims(gate, axis=1)
        print(gate.get_shape())
        task_experts = tf.stack(task_experts, axis=1)
        print(task_experts.get_shape())
        task_input = tf.squeeze(tf.matmul(gate, task_experts), axis=1)
        task_tower_inputs.append(task_input)

    return task_tower_inputs

def shared_embedding(dnn_inputs, args, mode):
    task_inputs = expert_layer([dnn_inputs] * args.task_num, args, mode, expert_num = args.task_num)
    return task_inputs

def shared_bottom(dnn_inputs, args, mode):
    task_inputs = expert_layer([dnn_inputs], args, mode, expert_num = 1)   
    return task_inputs * args.task_num    

def ple(dnn_inputs, args, mode):
    layer_inputs = [dnn_inputs] * args.expert_num
    layer_outputs = [dnn_inputs] * args.expert_num
    assert args.ple_level_cnt >= 1
    for k in range(args.ple_level_cnt):
        all_experts = expert_layer(layer_inputs, args, mode)
        layer_outputs = []
        for i in range(args.expert_num):
            if i < args.task_num:
                task_experts = [all_experts[i]]
                task_experts.extend([all_experts[j] for j in range(args.task_num, args.expert_num)])
            else:
                if k < args.ple_level_cnt - 1:
                    task_experts = [all_experts[j] for j in range(args.expert_num)]
                else: # do not need update shared experts at last layer
                    break
                
            gate = tf.layers.dense(layer_inputs[i], len(task_experts), activation=tf.nn.softmax)
            #tf.Print(gate, [gate], message="gate: ", first_n=10, summarize=5)
            gate = tf.expand_dims(gate, axis=1)
            task_experts = tf.stack(task_experts, axis=1)
            task_input = tf.squeeze(tf.matmul(gate, task_experts), axis=1)
            layer_outputs.append(task_input)
        layer_inputs = layer_outputs
            
    return layer_outputs[:args.task_num]

def l0_norm(target, temperature, lower, higher):
    """Returns the L0 norm."""
    offsets = temperature * tf.math.log(-lower / higher)
    dense_probs = tf.nn.sigmoid(target - offsets)
    # structure is sample independent, just reduce sum
    return tf.reduce_sum(dense_probs)

def hard_sigmoid(target, lower, higher):
    """Stretches and clips sigmoid samples between 0 and 1."""
    samples = tf.nn.sigmoid(target) * (higher - lower) + lower
    return tf.clip_by_value(samples, 0., 1.)

def sample(target, temperature, lower, higher, mode, eps=1e-20):
    U = tf.random.uniform(tf.shape(target, out_type=tf.int32), minval=0, maxval=1, dtype=tf.float32)
    sample = (tf.math.log(U + eps) - tf.math.log((1 - U) + eps) + target)/temperature
    if mode == tf.estimator.ModeKeys.TRAIN:
        sample = sample
    else:
        sample = target 
    return hard_sigmoid(sample, lower, higher)    

def mssm(dnn_inputs, args, mode, level_cnt, cell_level=True, task_specific=True):
    snr_l0_loss = 0
    if args.enable_fscm:
        uniq_feature_cnt = args.uniq_feature_cnt
        slot_cnt = len(uniq_feature_cnt)
        features = tf.split(dnn_inputs, num_or_size_splits=slot_cnt, axis=-1) 
        layer_inputs = []
        for i in range(args.expert_num):
            expert_inputs = []
            for j in range(len(features)):
                w_i_j = tf.layers.dense(features[j], 1, use_bias=False, activation=tf.nn.tanh)
                w_i_j = tf.nn.relu(w_i_j)
                expert_inputs.append(w_i_j * features[j])
            fscm_out = tf.layers.dense(tf.concat(expert_inputs, axis=-1), 
                                       args.expert_layers[-1],
                                       use_bias=False,
                                       activation=tf.identity)
            layer_inputs.append(fscm_out)
    else:
        layer_inputs = expert_layer([dnn_inputs]*args.expert_num, args, mode)        
    layer_outputs = []
    assert level_cnt >= 1
    for k in range(level_cnt):
        layer_outputs = []
        if k == level_cnt - 1:
            next_expert_num = args.task_num
        else:
            next_expert_num = args.expert_num
        for i in range(next_expert_num):
            if i < args.task_num and task_specific:
                task_inputs = [layer_inputs[i]]
                task_inputs.extend([layer_inputs[j] for j in range(args.task_num, args.expert_num)])
            else:
                task_inputs = [layer_inputs[j] for j in range(args.expert_num)]
            if cell_level:
                log_alpha = tf.get_variable('log_alpha_{0}_{1}'.format(k, i), 
                                            shape=(len(task_inputs), args.expert_layers[-1]), 
                                            initializer=tf.constant_initializer(-0.5),
                                            dtype=tf.float32)
            else:
                log_alpha = tf.get_variable('log_alpha_{0}_{1}'.format(k, i), 
                                            shape=(len(task_inputs), 1), 
                                            initializer=tf.constant_initializer(-0.5),
                                            dtype=tf.float32)

            snr_l0_loss += l0_norm(log_alpha, args.snr_temperature, args.snr_lower, args.snr_higher)
            route = sample(log_alpha, args.snr_temperature, args.snr_lower, args.snr_higher, mode) 
            route = tf.expand_dims(route, axis=0)
            if args.snr_mode == 'trans':
                out_to_upper = [tf.layers.dense(e, args.expert_layers[-1], 
                                           use_bias=False,
                                           activation=tf.identity) for e in task_inputs]
            else:
                out_to_upper = task_inputs
            out_to_upper = tf.stack(out_to_upper, axis=1)
            out_to_upper = out_to_upper * route
            net_input = tf.reduce_sum(out_to_upper, axis=1)
            layer_outputs.append(net_input) 
        layer_inputs = layer_outputs
    return layer_outputs, snr_l0_loss

def snr(dnn_inputs, args, mode):
    def _SNR(dnn_inputs):
        snr_l0_loss = 0
        layer_inputs = dnn_inputs
        layer_outputs = []
        assert args.snr_level_cnt >= 1
        for k in range(args.snr_level_cnt):
            all_experts = expert_layer(layer_inputs, args, mode)
            layer_outputs = []
            if k == args.snr_level_cnt - 1:
                next_expert_num = args.task_num
            else:
                next_expert_num = args.expert_num
            for i in range(next_expert_num):
                log_alpha = tf.get_variable('log_alpha_{0}_{1}'.format(k, i), 
                                                      shape=(1, args.expert_num), 
                                                      initializer=tf.keras.initializers.glorot_uniform(),
                                                      dtype=tf.float32)
                snr_l0_loss += l0_norm(log_alpha, args.snr_temperature, args.snr_lower, args.snr_higher)
                route = sample(log_alpha, args.snr_temperature, args.snr_lower, args.snr_higher, mode) 
                route = tf.expand_dims(route, axis=1)
                if args.snr_mode == 'trans':
                    all_experts = [tf.layers.dense(e, args.expert_layers[-1], 
                                                   use_bias=False,
                                                   activation=tf.identity) for e in all_experts]
                else:
                    assert args.snr_mode == 'aver'
                net_input = tf.stack(all_experts, axis=1)
                net_input = tf.squeeze(tf.matmul(route, net_input), axis=1)
                layer_outputs.append(net_input) 
            layer_inputs = layer_outputs
        return layer_outputs, snr_l0_loss
    dnn_inputs = dnn_inputs if isinstance(dnn_inputs, list) else [dnn_inputs] * args.expert_num
    task_tower_inputs, snr_l0_loss = _SNR(dnn_inputs) 
    return task_tower_inputs, snr_l0_loss
        
def snr_v1(dnn_inputs, args, mode):
    def l0_norm(target, temperature, lower, higher):
        """Returns the L0 norm."""
        offsets = temperature * tf.math.log(-lower / higher)
        dense_probs = tf.nn.sigmoid(target - offsets)
        return tf.reduce_mean(tf.reduce_sum(dense_probs, axis=-1, keepdims = True))
    def hard_sigmoid(target, lower, higher):
        """Stretches and clips sigmoid samples between 0 and 1."""
        samples = tf.nn.sigmoid(target) * (higher - lower) + lower
        return tf.clip_by_value(samples, 0., 1.)
    def sample(target, temperature, lower, higher, mode, eps=1e-20):
        U = tf.random.uniform(tf.shape(target, out_type=tf.int32), minval=0, maxval=1, dtype=tf.float32)
        sample = (tf.math.log(U + eps) - tf.math.log((1 - U) + eps) + target)/temperature
        if mode == tf.estimator.ModeKeys.TRAIN:
            sample = sample
        else:
            sample = target 
        return hard_sigmoid(sample, lower, higher)
    def _SNR(input_layer, snr_layers):
        snr_l0_loss = 0
        for i, pair in enumerate(snr_layers) :
            node_num = pair[0]
            net_num = pair[1]
            next_layer = []
            if i != 0:
                for j in range(net_num):
                    log_alpha = tf.compat.v1.get_variable('log_alpha_{0}_{1}'.format(i, j), shape=(1, route_num), initializer=tf.constant_initializer(-0.5), dtype=tf.float32)    
                    snr_l0_loss += l0_norm(log_alpha, config.SNR_temperature, config.SNR_lower, config.SNR_higher)
                    route = sample(log_alpha, config.SNR_temperature, config.SNR_lower, config.SNR_higher, mode) 
                    route = tf.expand_dims(route, axis=1)
                    if args.snr_mode == 'trans':
                        input_layer = [tf.layers.dense(net, node_num, 
                                                       use_bias=False,
                                                       activation=tf.identity) for net in input_layer]
                    else:
                        assert args.snr_mode == 'aver'
                    net_input = tf.stack(input_layer, axis=1)
                    net_input = tf.squeeze(tf.matmul(route, net_input), axis=1)
                    next_layer.append(net_input) 
                input_layer = next_layer
                route_num = net_num
            else:
                route_num = net_num
        return next_layer, snr_l0_loss
    all_experts = []
    for i in range(args.expert_num):
        expert_input = dnn_inputs
        for j, node_num in enumerate(args.expert_layers):
            expert_input = tf.layers.dense(expert_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                expert_input = tf.nn.dropout(expert_input, keep_prob=args.keep_prob[0])
        all_experts.append(expert_input)
    task_tower_inputs, snr_l0_loss = _SNR(all_experts, config.SNR_layers) 
    return task_tower_inputs, snr_l0_loss

def expert_layer(expert_inputs, args, mode, expert_num=None):
    all_experts = []
    if expert_num is None:
        expert_num = args.expert_num
    for i in range(expert_num):
        expert_input = expert_inputs[i]
        for j, node_num in enumerate(args.expert_layers):
            expert_input = tf.layers.dense(expert_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                expert_input = tf.nn.dropout(expert_input, keep_prob=args.keep_prob[0])
        all_experts.append(expert_input)    
    return all_experts

def gen_embeddings(feat_dnn, args, idx=0):
    uniq_feature_cnt = args.uniq_feature_cnt
    slot_cnt = len(uniq_feature_cnt)
    dnn_inputs = []
    feat_dnn = tf.split(feat_dnn, num_or_size_splits=slot_cnt, axis=-1)
    l2_reg = tf.keras.regularizers.l2(args.lamda)
    with tf.variable_scope("model"):
        for i in range(slot_cnt):
            vocab_size = uniq_feature_cnt[i]
            DEEP_V = tf.get_variable(name='emb_%d_%d' % (i, idx), 
                                     shape=[vocab_size, args.embedding_dim], 
                                     regularizer=l2_reg,
                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
                                     #initializer=tf.glorot_normal_initializer())
            feature_emb = tf.nn.embedding_lookup(DEEP_V, feat_dnn[i])
            dnn_inputs.append(feature_emb)

        dnn_inputs = tf.concat(dnn_inputs, axis=-1) 
        dnn_inputs = tf.squeeze(dnn_inputs, axis=-2)
    return dnn_inputs
 
def single_task_model(dnn_inputs, args, mode):
    task_outputs = []
    for i in range(args.task_num):
        task_input = dnn_inputs[i]
        for j, node_num in enumerate(args.expert_layers + args.task_layers):
            task_input = tf.layers.dense(task_input, node_num, activation=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task_input = tf.nn.dropout(task_input, keep_prob=args.keep_prob[1])

        task_out = tf.layers.dense(task_input, 1, activation=tf.identity, use_bias=True)
        task_outputs.append(task_out)
    return task_outputs, 0.0
                       
def model_fn(features, labels, mode=None, params=None):
 
    args = params['args']
    feat_dnn, size = features['feat'],  features['size']
    task_labels = []
    for i in range(args.task_num):
        task_labels.append(features['label%d' % (i)])

    if args.model != 'st':
        print('train multi-task models with shared embeddings')
        dnn_inputs =  gen_embeddings(feat_dnn, args, 0)
        task_outputs, snr_l0_loss = model(dnn_inputs, args, mode)
    else:
        print('train single task models with separate embeddings')
        dnn_inputs = [gen_embeddings(feat_dnn, args, i)
                      for i in range(args.task_num)]  
        task_outputs, snr_l0_loss = single_task_model(dnn_inputs, args, mode)     
        
    task_preds = []
    for i in range(args.task_num):
        if args.task_loss[i] == 'xent':
            pred = tf.sigmoid(task_outputs[i])
        else:
            pred = task_outputs[i]
        task_preds.append(pred)
    
    predictions = {}
    for i in range(args.task_num):
        predictions['task%d'%i]=task_preds[i]
        predictions['task%d_l'%i]=task_labels[i]

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
    loss = 0
    for i in range(args.task_num):
        if args.task_loss[i] == 'xent':
            loss = loss + tf.losses.log_loss(task_labels[i], task_preds[i])
        elif args.task_loss[i] == 'mse':
            loss = loss + tf.losses.mean_squared_error(task_labels[i], task_preds[i])
        else:
            assert False
 
    reg_variables = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    if args.lamda > 0:
        reg_loss = tf.add_n(reg_variables)
    else:
        reg_loss = 0
    loss += reg_loss
    loss += args.snr_l0_loss_weight * snr_l0_loss
    # Provide an estimator spec for `ModeKeys.EVAL`
    ROC_bucket = 20000
    eval_metric_ops = {}
    metric_sum = 0.0
    for i in range(args.task_num):
        if args.task_loss[i] == 'xent':
            eval_metric_ops['auc_%d'%i] = tf.metrics.auc(task_labels[i], task_preds[i], num_thresholds=ROC_bucket, curve='ROC')
            metric_sum = metric_sum + eval_metric_ops['auc_%d'%i][0] * args.task_weight[i]
        else:
            eval_metric_ops['mse_%d'%i] = tf.metrics.mean_squared_error(task_labels[i], task_preds[i])
            metric_sum = metric_sum + eval_metric_ops['mse_%d'%i][0] * args.task_weight[i]
    eval_metric_ops['metric_sum'] = (metric_sum, tf.no_op())
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
    else:  # 'sgd'
        optimizer = tf.train.GradientDescentOptimizer(args.lr)
    print(args.optimizer, optimizer)
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






