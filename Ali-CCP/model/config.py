# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys

num_threads = 10

expert_num = 3
expert_layers = [128]
task_num = 2
task_experts = [
    (0, 1, 2),
    (0, 1, 2)
]
task_experts_ple = [
    (0, 1),
    (1, 2)
]
task_layers = [80]

#v1
dnn_layers = [200, 80]




