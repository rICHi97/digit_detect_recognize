# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:43:45 2021

@author: rICHi0923
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
from sklearn.cluster import DBSCAN

project_dir = os.path.abspath('../..')
sys.path.append(project_dir)
from data_process.clustering_regression_recs_correction import _get_rec_center
from data_process.clustering_regression_recs_correction import *

image_dir = '../../image/'
image_txt_dir = '../../results/east_detect/image_txt/'
results_dir = '../../results/data_process/PCA_optimize_results/'
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
    if img_name != '1':
        continue
    img_recs_txt = img_name + '_original.txt'
    all_recs = []
    with open(image_txt_dir + img_recs_txt) as recs_txt:
        lines = recs_txt.readlines()
        for i in range(len(lines)):
            line = lines[i].split(',')
            rec = [float(x) for x in line]
            # all_recs之前没有清空
            all_recs.append(rec)
        pca_, new_rec_data_list = PCA_(all_recs, width, height)
        new_rec_data_array = np.array(new_rec_data_list).reshape((-1, 1))
        start = time.clock()
        tdt, rec_group_list, avg_after_filter, std = divide_rec_list(new_rec_data_list)
            # labels = DBSCAN(eps = 1.5 * (max(new_rec_data_list) - min(new_rec_data_list)) / len(lines), min_samples = 3).fit(new_rec_data_array).labels_

        # rec_i - rec_i-1， cnt - 1个数据
        # delta_one_rec_data_list = get_delta_rec_list(1, new_rec_data_list)
        # rec_i - rec_i-2， cnt - 2个数据
        # delta_two_rec_data_list = get_delta_rec_list(2, new_rec_data_list)
        # avg_delta_one_rec = sum(delta_one_rec_data_list) / len(delta_one_rec_data_list)
        # avg_delta_two_rec = sum(delta_two_rec_data_list) / len(delta_two_rec_data_list)
        # tdt, rec_group_list, avg_after_filter, std = divide_rec_list(new_rec_data_list)

        # draw.text((0, 0), '%f'%(avg_delta_one_rec), fill = 'black', font = draw_font)
        # draw.text((100, 0), '%f'%(avg_after_filter), fill = 'blue', font = draw_font)
        # draw.text((200, 0), '%f'%(std), fill = 'red', font = draw_font)
        # draw.text((200, 200), '%s'%(tdt), fill = 'yellow', font = draw_font)
        
        '''
        for i in range(len(rec_group_list)):
            rec_group = rec_group_list[i]
            for j in rec_group:
                rec = all_recs[j]
                center = _get_rec_center(rec)
                draw_position = (center[0] - 3 * font_width, center[1])
                draw.text(draw_position, '%.4f')
                # draw.text(draw_position, '第%s类'%(i), fill =  'black' ,font = draw_font)
        '''
        # for i in range(len(lines)):
        #     rec = all_recs[i]
        #     center = _get_rec_center(rec)
        #     draw_position = (center[0] - 3 * font_width, center[1])
        #     # draw.text(draw_position, '%.4f'%(new_rec_data_list[i]), fill =  'black' ,font = draw_font)
        #     draw.text(draw_position, '第%d类'%(labels[i]), fill =  'black' ,font = draw_font)
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
        