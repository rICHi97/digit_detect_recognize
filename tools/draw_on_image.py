# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:28:45 2021

@author: rICHi0923
"""
from PIL import Image, ImageDraw
from math import sin, cos
import os
import numpy as np
from data_process.clustering_regression_recs_correction import get_four_point_by_rec_data
img_dir = '../east/test/image/'
txt_dir = '../east/test/image_txt/'
results_dir = '../results/'

# 用于在rec中显示起始边
first_side_flag = False

# 通过rec四点坐标画rec
def draw_rec(rec, img, color):
    draw = ImageDraw.Draw(img)
    rec = [float(x) for x in rec]
    if first_side_flag:
        if color == 'yellow':
            draw.line([tuple([rec[0], rec[1]]),
                       tuple([rec[2], rec[3]])], width = 2, fill = 'blue')
        else: 
            draw.line([tuple([rec[0], rec[1]]),
                       tuple([rec[2], rec[3]])], width = 2, fill = 'yellow')
    else:
        draw.line([tuple([rec[0], rec[1]]),
                   tuple([rec[2], rec[3]])], width = 2, fill = color)
    draw.line([tuple([rec[2], rec[3]]),
               tuple([rec[4], rec[5]]),
               tuple([rec[6], rec[7]]),
               tuple([rec[0], rec[1]])], width = 2, fill = color)
    
# 通过rec的中心坐标、平均长度、平均旋转角画rec
# rec_data = [center_x, center_y, new_length_W, new_length_H, new_rotate_angle_W, new_rotate_angle_H1, new_rotate_angle_H2]
def draw_rec_by_rec_data(rec_data, img, color):
    rec = get_four_point_by_rec_data(rec_data)
    draw_rec(rec, img, color) 

# rec_type = corrected/original
# 返回一张图片
def draw_recs_by_txt(txt_path, img_path, color):
    img = Image.open(img_path)
    img_copy = img.copy()
    with open(txt_path) as txt:
        lines = txt.readlines()
        for i in range(len(lines)):
            rec = lines[i].split(',')
            draw_rec(rec, img_copy, color)
        # img_copy.save(out_path + '/%s.jpg'%(img_name))
    return img_copy

# 在一张图上画corrected和original recs以用来对比
# 先加载原始图片，画original_rec框，然后加载画了original_rec框的图片画corrected_rec框
# name为不包括格式的图片名，设置为'all'会画全部的图      
def draw_corrected_and_original_recs_on_one_img(name):
    for img_file in os.listdir(img_dir):
        img_name = img_file.split('.')[0]
        if img_name == name or name == 'all':
            original_txt_path = txt_dir + '%s_original.txt'%(img_name) 
            corrected_txt_path = txt_dir + '%s_corrected.txt'%(img_name)
            original_recs_img = draw_recs_by_txt(original_txt_path, img_dir + '%s'%(img_file), 'blue')
            original_recs_img_path = results_dir + 'original_recs/%s_original.jpg'%(img_name)
            original_recs_img.save(original_recs_img_path)
            corrected_original_recs_img = draw_recs_by_txt(corrected_txt_path, original_recs_img_path, 'yellow')
            corrected_original_recs_img_path = results_dir + 'corrected_original_recs/%s_corrected_original.jpg'%(img_name)
            corrected_original_recs_img.save(corrected_original_recs_img_path)
        else:
            continue

            
        
        

