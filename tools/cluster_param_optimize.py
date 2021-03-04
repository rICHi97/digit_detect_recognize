# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:08:03 2021

@author: rICHi0923
"""
import sys
import os
project_dir = os.path.abspath('../')
sys.path.append(project_dir)
from PIL import Image, ImageDraw, ImageFont
from data_process.clustering_regression_recs_correction import DBSCAN_, _get_rec_center

image_dir = '../results/矫正样本/'
image_txt_dir = '../east/test/image_txt/'
results_dir = '../results/DBSCAN_optimize_results/'
cluster_txt = '10147_original.txt'
# img_name = lof_txt.split('.')[0]

recs_all = []


font = ImageFont.truetype(font = 'simhei.ttf', size = 20)
font_width = font.getsize('1')[0]
imgs = os.listdir(image_dir)
control_times = 999
cnt = 0
for _ in imgs:
    img = Image.open(image_dir + '%s'%(_))
    width, height = img.width, img.height
    draw = ImageDraw.Draw(img)
    img_name = _.split('.')[0]
    img_recs_txt = img_name + '.txt'
    recs_all = []
    with open(image_txt_dir + img_recs_txt) as recs_txt:
        lines = recs_txt.readlines()
        for i in range(len(lines)):
            line = lines[i].split(',')
            rec = [float(x) for x in line]
            # recs_all之前没有清空
            recs_all.append(rec)
        labels, rec_data = DBSCAN_(recs_all, width, height)    
        
        for i in range(len(lines)):
            rec = recs_all[i]
            center = _get_rec_center(rec)
            draw_position = (center[0] - 3 * font_width, center[1])
            if len(labels) == len(lines):
                draw.text(draw_position, '第%s类'%(labels[i]), fill =  'black' ,font = font)
            # 指示中心
            # draw.rectangle((center[0], center[1], center[0] + 5, center[1] + 5), fill = 'yellow')
    img.save(results_dir + '%s_DBSCAN_result.%s'%(img_name, _.split('.')[-1]))
    
    cnt += 1
    if cnt >= control_times:
        break
    if img_name == 'terminal_13_original':
        break

     
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
        