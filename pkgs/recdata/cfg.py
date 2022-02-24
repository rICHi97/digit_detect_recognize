# -*- coding: utf-8 -*-
"""
Created on 2021-12-11 02:54:57

@author: Li Zhi
"""
# recdata_processing
W_coef = 0.25
H_coef = 0.37
offset_coef = -0.22
joint_img_dir = './resource/joint_img/'

# recdata_correcting
# Correction
min_terminal_cnt_to_correct = 3 # 大于阈值个数的端子组才会矫正
reg_indep_vars = ['center'] # 自变量
reg_dep_vars = ['length_W', 'length_H', 'rotate_angle_W'] # 因变量
