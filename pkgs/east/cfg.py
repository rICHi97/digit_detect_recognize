# -*- coding: utf-8 -*-
"""
Created on 2021-11-17 15:59:15

@author: Li Zhi
"""
train_task_id = '3T832'
initial_epoch = 0
epoch_num = 48

data_dir = './resource/train_data/'
origin_img_dir = 'a_img'
origin_txt_dir = 'a_txt'
train_img_dir = 'b_train_img'
train_label_dir = 'b_train_label'
preprocess_img_dir = 'c_preprocess_img'
label_img_dir = 'c_label_img'

save_weights_filepath = f'./resource/east_model/{train_task_id}.h5'
east_weights_filepath = './resource/east_model/019-0.142.h5'
# east_weights_filepath = './resource/east_model/weights_3T832.020-0.113.h5'
img_dir = './resource/test_data/image/'
predict_img_dir = None
output_txt = True
output_txt_dir = './resource/test_data/image_txt'

val_ratio = 0.2
val_filename = f'val_{train_task_id}.txt'
train_filename = f'train_{train_task_id}.txt'
total_img = 14
batch_size = 2  # batch_size应该随img_size而调整
steps_per_epoch = total_img * (1 - val_ratio) // batch_size
val_steps = total_img * val_ratio // batch_size
summary = True
train_verbose = True
lr = 1e-4
decay = 5e-4

preprocess_verbose = False
show_preprocess_img = False
show_label_img = False
show_predict_img = False

# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.15  # 原始为0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.3 # 原始为0.6
epsilon = 1e-4

max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])
num_channels = 3
num_img = 1
# locked_layers = True  # 测试，这个lock layers好像无用
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]  # pixel_size = 4
pixel_threshold = 0.9 # 原始为0.9，越大越严格
side_vertex_pixel_threshold = 0.8  # 原始为0.8，越大越严格，判断是否为内部像素
trunc_threshold = 0.2 # 原始为0.2，越小越严格，判断头尾像素

# 控制三个loss的系数
lambda_class_score_loss = 4.0
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

callbacks = ['early_stopping', 'check_point', 'reduce_lr']
early_stopping_patience = 8
early_stopping_verbose = True
check_point_filepath = './resource/east_model/{epoch:03d}-{val_loss:.3f}.h5'
check_point_period = 1
check_point_verbose = True
reduce_lr_monitor = 'val_loss'
reduce_lr_factor = 0.1
reduce_lr_patience = 4
reduce_lr_verbose = True
reduce_lr_min_lr = 1e-6
