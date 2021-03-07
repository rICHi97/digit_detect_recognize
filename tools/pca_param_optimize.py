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
from data_process.clustering_regression_recs_correction import PCA_, _get_rec_center, get_delta_rec_list, divide_rec_list
import numpy as np
import time
image_dir = '../results/矫正样本/'
image_txt_dir = '../east/test/image_txt/'
results_dir = '../results/PCA_optimize_results/'
cluster_txt = '10147_original.txt'
# 两列/一列一类/一列多类
# terminal_distribution_types = ('2cols', '1col_1class', '1col_nclasses')
       
# img_name = lof_txt.split('.')[0]
all_recs = []
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
    img_recs_txt = img_name + '.txt'
    all_recs = []
    with open(image_txt_dir + img_recs_txt) as recs_txt:
        lines = recs_txt.readlines()
        for i in range(len(lines)):
            line = lines[i].split(',')
            rec = [float(x) for x in line]
            # all_recs之前没有清空
            all_recs.append(rec)
        pca_, new_rec_data_list = PCA_(all_recs, width, height)
        # rec_i - rec_i-1， cnt - 1个数据
        # delta_one_rec_data_list = get_delta_rec_list(1, new_rec_data_list)
        # rec_i - rec_i-2， cnt - 2个数据
        # delta_two_rec_data_list = get_delta_rec_list(2, new_rec_data_list)
        # avg_delta_one_rec = sum(delta_one_rec_data_list) / len(delta_one_rec_data_list)
        # avg_delta_two_rec = sum(delta_two_rec_data_list) / len(delta_two_rec_data_list)
        tdt, rec_group_list, avg_after_filter, std = divide_rec_list(new_rec_data_list)

        # draw.text((0, 0), '%f'%(avg_delta_one_rec), fill = 'black', font = draw_font)
        draw.text((100, 0), '%f'%(avg_after_filter), fill = 'blue', font = draw_font)
        draw.text((200, 0), '%f'%(std), fill = 'red', font = draw_font)
        draw.text((200, 200), '%s'%(tdt), fill = 'yellow', font = draw_font)

        for i in range(len(rec_group_list)):
            rec_group = rec_group_list[i]
            for j in rec_group:
                rec = all_recs[j]
                center = _get_rec_center(rec)
                draw_position = (center[0] - 3 * font_width, center[1])
                draw.text(draw_position, '第%s类'%(i), fill =  'black' ,font = draw_font)

        # for i in range(len(lines)):
        #     rec = all_recs[i]
        #     center = _get_rec_center(rec)
        #     draw_position = (center[0] - 3 * font_width, center[1])
        #     # draw.text(draw_position, '%.4f'%(new_rec_data_list[i]), fill =  'black' ,font = draw_font)
        #     if i > 0:
        #         draw.text((draw_position[0], draw_position[1]), '%.4f'%(delta_one_rec_data_list[i - 1] / avg_after_filter), fill =  'black' ,font = draw_font)
        #     if i > 1:
        #        draw.text(draw_position, '%.4f'%(delta_two_rec_data_list[i - 2]), fill =  'blue' ,font = draw_font)
    img.save(results_dir + '%s_PCA_result.%s'%(img_name, _.split('.')[-1]))
    
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
#         all_recs.append(rec)
#     factors, results, rec_data_list = LOF(all_recs)    
    
#     for i in range(len(lines)):
#         rec = all_recs[i]
#         center = _get_rec_center(rec)
#         draw.text(center, '%.4f'%(factors[i]), fill = 'gray' ,font = font)  
# img.save('./%s_lof_result.jpg'%(img_name))
        