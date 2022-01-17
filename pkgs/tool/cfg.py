# -*- coding: utf-8 -*-
"""
Created on 2021-12-07 15:40:09

@author: Li Zhi
"""
# 文件路径基准目录是入口文件，即主目录中的test.py

# code_testing
test_joint_rec_img_path = './resource/test_data/image/1.jpg'
test_joint_rec_txt_path = './resource/test_data/image_txt/1.txt'
test_get_text_area_img_path = './resource/test_data/image/10148.png'
test_get_text_area_txt_path = './resource/10148.txt'
test_recognize_img_path = './resource/test_data/image/terminal_5_number_1.jpg'
test_recognize_recs_txt_path = './resource/test_data/label_txt/terminal_5_number_1.txt'
test_draw_label_txt_img_path = './resource/train_data/image/terminal_10_number_1.jpg'
test_draw_label_txt_label_txt_path = './resource/train_data/label_txt/terminal_10_number_1.txt'
# 0 1 10 13 15 17
test_draw_gt_gt_path = './resource/train_data/b_train_label/terminal_10_number_13_gt.npy' # 修改后label文件
# test_draw_gt_gt_path = './resource/test_data/gt_image/terminal_10_number_10_gt.npy' # 之前版本label
test_draw_gt_img_path = './resource/test_data/gt_image/terminal_10_number_13.jpg'
test_draw_gt_resized = False
test_end_to_end_img_path = './resource/test_data/image/10148.png'

# image_processing
coef_x_len, coef_y_len = 0.23, 0.02
joint_img_dir = './resource/joint_img/'
max_joint_img_width = 1000
joint_img_height = 60
img_rec_height = 40
spacing = 30
threshold = True

# tool
# UiTool
# Ui_window_name.py中使用绝对导入qrc_path
# 作为非入口文件时，其顶层目录为入口文件目录，也就是文件夹的根目录
qrc_path = './'
qrc_output_dir = './pkgs/ui/qt_designer_code/'
delete_qrc = False
old_code_dir = './pkgs/ui/qt_designer_code/'
new_code_dir = r'D:\GitHub_Project\ERIC_WORKSPACE\terminal_detect_recognize'
update_keyword = 'Ui'

# visualization
gt_max_img_size = 832
gt_output_dir = './resource/test_data/'
