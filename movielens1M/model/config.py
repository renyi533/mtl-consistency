# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys

num_threads = 10

batch_size = 100
optimizer = 'sgd'
learning_rate = 0.01
dropout = 1.0

feature_num = 7
feature_size_dnn = 20000
feature_size_title = 5000
feature_size_genres = 20

embedding_size_dnn = 24

expert_num = 3
expert_layers = [256, 128]
task_num = 2
task_experts = [
    (0, 1, 2),
    (0, 1, 2)
]
task_layers = [64]




