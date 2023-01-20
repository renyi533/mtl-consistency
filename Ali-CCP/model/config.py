# -*- coding: utf-8 -*-
# yingdu @ 2023-01-05

import sys

num_threads = 10

SNR_lower = -1.0
SNR_higher = 3.0 
SNR_temperature = 1.0
SNR_layers = [(128,3), (128,3), (128,2)] # (node_num, net_num)