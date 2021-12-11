# -*- coding: utf-8 -*-
"""
Created on 2021-12-07 15:40:09

@author: Li Zhi
"""


# code_testing
# 文件路径基准目录是入口文件，即主目录中的test.py
test_joint_rec_img_path = './source/test_data/image/1.jpg'
test_joint_rec_txt_path = './source/test_data/image_txt/1.txt'
test_recognize_img_path = './source/test_data/image/terminal_5_number_1.jpg'
test_recognize_recs_txt_path = './source/test_data/label_txt/terminal_5_number_1.txt'

# image_processing
coef_x_len, coef_y_len = 0.23, 0.02
joint_img_dir = './source/joint_img/'
max_joint_img_width = 1000
joint_img_height = 42
img_rec_height = 30
spacing = 20
preprocess_img_rec = True
