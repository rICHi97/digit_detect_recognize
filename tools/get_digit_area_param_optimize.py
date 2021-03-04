# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:43:45 2021

@author: rICHi0923
"""

import sys
import os
project_dir = os.path.abspath('../')
sys.path.append(project_dir)
from PIL import Image, ImageDraw, ImageFont
from data_process.clustering_regression_recs_correction import _get_rec_center
from data_process.clustering_regression_recs_correction import *
# import data_process.clustering_regression_recs_correction
from draw_on_image import draw_rec
import numpy as np
import time
image_dir = '../results/矫正样本/'
image_txt_dir = '../east/test/image_txt/'
results_dir = '../results/get_digit_area_optimize_results/'
   
# img_name = lof_txt.split('.')[0]
recs_all = []
draw_font = ImageFont.truetype(font = 'simhei.ttf', size = 20)
font_width, font_height = draw_font.getsize('1')[0], draw_font.getsize('1')[1]
imgs = os.listdir(image_dir)
control_times = 999
cnt = 0
start = time.clock()
for _ in imgs:
    img = Image.open(image_dir + '%s'%(_))
    width, height = img.width, img.height
    draw = ImageDraw.Draw(img)
    img_name = _.split('.')[0]
    # if img_name != '1_original':
    #     continue
    img_recs_txt = img_name + '.txt'
    recs_all = []
    with open(image_txt_dir + img_recs_txt) as recs_txt:
        lines = recs_txt.readlines()
        for i in range(len(lines)):
            line = lines[i].split(',')
            rec = [float(x) for x in line]
            # recs_all之前没有清空
            recs_all.append(rec)
        new_recs_all = recs_correction(recs_all, width, height)
        digit_areas = get_all_digit_areas(new_recs_all)
        for rec in digit_areas:
            draw_rec(rec, img, 'yellow')
    img.save(results_dir + '%s_digit_area.jpg'%(img_name))
    
    cnt += 1
    if cnt >= control_times:
        break

end = time.clock()     
# img = Image.open(image_dir + '%s.jpg'%(img_name))
# draw = ImageDraw.Draw(img)
# with open(image_txt_dir + cluster_txt
#           ) as recs_txt:
#     lines = recs_txt.readlines()
#     for i in range(len(lines)):
#         line = lines[i].split(',')
#         rec = [float(x) for x in line]
#         recs_all.append(rec)
#     factors, results, rec_data_list = LOF(recs_all)    
    
#     for i in range(len(lines)):
#         rec = recs_all[i]
#         center = _get_rec_center(rec)
#         draw.text(center, '%.4f'%(factors[i]), fill = 'gray' ,font = font)  
# img.save('./%s_lof_result.jpg'%(img_name))
        