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

east_weights_file_path = './source/east_model/weights_3T832.020-0.113.h5'
img_dir = './source/test_data/image/'
predict_img_dir = None
output_txt_dir = './source/test_data/image_txt'

val_filename = f'val_{train_task_id}.txt'
train_filename = f'train_{train_task_id}.txt'

show_preprocess_img = True
show_label_img = True
show_predict_img = False

# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.15  # 原始为0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.3 # 原始为0.6
val_ratio = 0.1
epsilon = 1e-4

max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])
num_channels = 3
num_img = 1
locked_layers = False
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]  # pixel_size = 4
pixel_threshold = 0.7 # 原始为0.9，越大越严格
side_vertex_pixel_threshold = 0.6  # 原始为0.8，越大越严格，判断是否为内部像素
trunc_threshold = 0.1 # 原始为0.2，越小越严格，判断头尾像素
