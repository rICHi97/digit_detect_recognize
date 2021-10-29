# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:24:20 2021

@author: LIZHi
"""
import time

import numpy as np
from PIL import Image, ImageFont

import recdata_correcting, recdata_processing, recdata_io
import visualization

# TODO：矫正后端子四点坐标不对
# TODO：读取一个文件夹中的txt文件，在对应的图片上绘制需要进一步封装
# TODO：选取三种类型的端子各一张进行测试，可以将矫正shape data与原始shape data比对
#       若相差大，怀疑是函数（从shape data得到xy list）有问题
# 1.jpg：双列 2.png：单列单线 10141，jpg：单列单线

test_correct_all_imgs = False

start = time.process_time()
txt_name, img_name = '2_original.txt', '2.png'
img_test_name = '2_test.jpg'
recs_xy_list = recdata_io.RecdataIO.read_rec_txt(txt_name)
original_recs_shape_data = []
for xy_list in recs_xy_list:
    rec_shape_data = recdata_processing.Recdata.get_rec_shape_data(xy_list)
    original_recs_shape_data.append(rec_shape_data)
img = Image.open(img_name).copy()
corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
# visualization.ImgDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
# visualization.ImgDraw.draw_recs(original_recs_shape_data, img, 2, 'black', True)
visualization.ImgDraw.draw_recs(corrected_recs_shape_data, img, 2, 'black', True)
img.save(img_test_name)

# 矫正多张图片
if test_correct_all_imgs:
    imgs_rec_dict = recdata_io.RecdataIO.read_rec_txt_dir('test/image_txt', keyword='original')
    i = 0
    imgs_xy_list = {}
    for key, recs_xy_list in imgs_rec_dict.items():
        img_name = key.split('original')[0][:-1]
        try:
            img = Image.open('test/image/' + img_name + '.jpg')
        except FileNotFoundError:
            img = Image.open('test/image/' + img_name + '.png')
        if len(recs_xy_list) < 3:
            i += 1
        else:
            corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
            _ = []
            for rec_shape_data in corrected_recs_shape_data:
                xy_list = recdata_processing.Recdata.get_xy_list(rec_shape_data)
                visualization.ImgDraw.draw_rec(
                    xy_list, img, width=2, color='black', distinguish_first_side=True
                    
                )
                _.append(xy_list)
            imgs_rec_dict[key] = _
        img.save('test/' + img_name + '.jpg')

end = time.process_time()
print(end - start)