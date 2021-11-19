# -*- coding: utf-8 -*-
"""
Created on 2021-11-17 15:59:15

@author: Li Zhi
"""
train_task_id = '3T832'

data_dir = './source/train_data/'
origin_img_dir = 'a_img'
origin_txt_dir = 'a_txt'
train_img_dir = 'b_train_img'
train_label_dir = 'b_train_label'
preprocess_img_dir = 'c_preprocess_img'
label_img_dir = 'c_label_img'

val_filename = f'val_{train_task_id}.txt'
train_filename = f'train_{train_task_id}.txt'

show_preprocess_img = True
show_label_img = True

# in paper it's 0.3, maybe to large to this problem
# 原始为0.2
shrink_ratio = 0.15
# pixels between 0.2 and 0.6 are side pixels
# 原始为0.6
shrink_side_ratio = 0.3
val_ratio = 0.1
epsilon = 1e-4

max_train_img_size = int(train_task_id[-3:])
num_channels = 3
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]  # pixel_size = 4