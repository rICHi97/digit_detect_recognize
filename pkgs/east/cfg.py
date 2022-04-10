# -*- coding: utf-8 -*-
"""
Created on 2021-11-17 15:59:15

@author: Li Zhi
"""
# format: {month:2}{day:2}T{train_num:2}{size:3}
# train_task_id不能随便设置，因为和preprocess、label生成txt文件有关
train_task_id = '0402T01512'
initial_epoch = 0
epoch_num = 128

data_dir = './resource/train_data/'
origin_img_dir = 'a_img'
origin_txt_dir = 'a_txt'
train_img_dir = 'b_train_img'
train_label_dir = 'b_train_label'
preprocess_img_dir = 'c_preprocess_img'
label_img_dir = 'c_label_img'

save_weights_filepath = f'./resource/east_model/{train_task_id}.h5' # 保存
vgg_pretrained_weights_filepath = './resource/east_model/126-0.068.h5' # vgg预训练
pva_pretrained_weights_filepath = './resource/east_model/087-0.070.h5' # pva预训练
inception_res_pretrained_weights_filepath = './resource/east_model/105-0.069.h5' # ir预训练
bd_east_pretrained_weights_filepath = './resource/east_model/011-0.111.h5/' # 双向预训练
east_weights_filepath = './resource/east_model/115-0.073.h5' # 加载
img_dir = './resource/test_data/image/'
predict_img_dir = None
output_txt = True
output_txt_dir = './resource/test_data/image_txt'

val_ratio = 0.1
val_filename = f'val_{train_task_id}.txt'
train_filename = f'train_{train_task_id}.txt'
total_img = 5828
batch_size = 24  # batch_size应该随img_size而调整
steps_per_epoch = total_img * (1 - val_ratio) // batch_size
val_steps = total_img * val_ratio // batch_size
summary = True
train_verbose = True
lr = 1e-2
fine_tune_lr = 1e-5
decay = 5e-4

preprocess_verbose = False
show_preprocess_img = False
show_label_img = False
show_predict_img = False

shrink_ratio = 0.15  # 原始为0.2
shrink_side_ratio = 0.3 # 原始为0.6，shrink_side_ratio与shrink_ratio之间是边界
epsilon = 1e-4

max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])
num_channels = 3
num_img = 1
pixel_size = 4
pixel_threshold = 0.9 # 原始为0.9，越大越严格
side_vertex_pixel_threshold = 0.8  # 原始为0.8，越大越严格，判断是否为内部像素
trunc_threshold = 0.2 # 原始为0.2，越小越严格，判断头尾像素

# 控制三个loss的系数
lambda_class_score_loss = 2.0
lambda_inside_score_loss = 2.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

callbacks = ['check_point', 'reduce_lr', 'tensorboard']
early_stopping_patience = 16
early_stopping_verbose = True
check_point_filepath = './resource/east_model/{epoch:03d}-{val_loss:.3f}.h5'
reduce_lr_monitor = 'val_loss'
reduce_lr_factor = 0.1
reduce_lr_patience = 16
reduce_lr_verbose = True
reduce_lr_min_lr = 1e-4
reduce_lr_min_fine_tune_lr = 1e-7
tensorboard_log_dir = './resource/train_data/logs/'
tensorboard_write_graph = False
tensorboard_update_freq = 12
