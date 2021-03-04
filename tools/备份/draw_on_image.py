# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:28:45 2021

@author: rICHi0923
"""
from PIL import Image, ImageDraw
import os

img_dir = '../east/test/image/'
txt_dir = '../east/test/image_txt/'
results_dir = '../results/'

# 用于在rec中显示起始边
first_side_flag = True

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
        draw.line(tuple(rec[0], rec[1]), width = 2, fill = color)
    draw.line([tuple([rec[2], rec[3]]),
               tuple([rec[4], rec[5]]),
               tuple([rec[6], rec[7]]),
               tuple([rec[0], rec[1]])], width = 2, fill = color)
    
    

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

            
        
        

