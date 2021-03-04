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
from data_process.clustering_regression_recs_correction import LOF, PCA_, get_delta_rec_list, divide_rec_list, _get_rec_center, regression_
import numpy as np
import time
image_dir = '../results/矫正样本/'
image_txt_dir = '../east/test/image_txt/'
results_dir = '../results/regression_optimize_results/'
        
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
    if img_name != '1_original':
         continue
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
        delta_one_rec_data_list = get_delta_rec_list(1, new_rec_data_list)
        # rec_i - rec_i-2， cnt - 2个数据
        delta_two_rec_data_list = get_delta_rec_list(2, new_rec_data_list)
        avg_delta_one_rec = sum(delta_one_rec_data_list) / len(delta_one_rec_data_list)
        avg_delta_two_rec = sum(delta_two_rec_data_list) / len(delta_two_rec_data_list)
        # 对rec分组，返回每组rec索引
        tdt, rec_group_list, avg_after_filter, std = divide_rec_list(avg_delta_one_rec, avg_delta_two_rec, new_rec_data_list)

        new_all_recs = all_recs
        for i in range(len(rec_group_list)):
            rec_group = rec_group_list[i]
            if len(rec_group) < 3:
                # 个数太少，不进行数据分析
                continue
            else:
                # 生成当前group中所有rec的坐标
                rec_data_group = [all_recs[x] for x in rec_group]
                # 筛选异常点
                factors, _, _ = LOF(rec_data_group)
                j = 0
                recs_after_LOF = []
                for rec in rec_data_group:
                    if abs(factors[j]) < 1.6:
                        recs_after_LOF.append(rec)
                    j += 1                  
                # 对剩下的点最小二乘回归，计算W长度与倾斜角W随x，y的关系
                regression_length_W, regression_rotate_angle_W = regression_(recs_after_LOF)

    img.save(results_dir + '%s_PCA_result.jpg'%(img_name))
    
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
        