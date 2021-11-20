# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:24:20 2021

@author: LIZHi
"""
import time
import os
import os.path as path

import numpy as np
from PIL import Image, ImageFont

from pkgs.recdata import recdata_correcting, recdata_io, recdata_processing
from pkgs.tool import image_processing, visualization
from pkgs.east import east_data, east_net

EastPreprocess = east_data.EastPreprocess
EastNet = east_net.EastNet
# TODO：矫正后端子四点坐标不对
# TODO：读取一个文件夹中的txt文件，在对应的图片上绘制需要进一步封装
# TODO：选取三种类型的端子各一张进行测试，可以将矫正shape data与原始shape data比对
#       若相差大，怀疑是函数（从shape data得到xy list）有问题
# 1.jpg：双列 2.png：单列单线 10141，jpg：单列单线


test_correct_all_imgs = False
test_correct_one_img = False
test_merge_json = False
test_crop_img = False
test_label = False
test_east_data = True

start = time.process_time()

img_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\img').replace('\\', '/')
label_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\txt_合并').replace('\\', '/')
output_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\裁切结果').replace('\\', '/')
json1_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\json').replace('\\', '/')
json2_dir = path.normpath(
    r'D:\各种文件\图像识别\端子排数据\标注整个边框\json_标注铭牌'
).replace('\\', '/')
# output_dir = path.normpath(
#     r'D:\各种文件\图像识别\端子排数据\标注整个边框\json_合并'
# ).replace('\\', '/')
if test_east_data:
        
      # EastPreprocess.preprocess()
      east = EastNet()
      east.predict()
                                             
if test_label:

    label_files = os.listdir(label_dir)
    for file in label_files:
        label_path = path.join(label_dir, file)
        img_name = file.replace('.txt', '.jpg')
        img_path = path.join(img_dir, img_name)
        if not path.exists(img_path):
            img_name = file.replace('.txt', '.png')
            img_path = path.join(img_dir, img_name)
        img = Image.open(img_path)
        visualization.ImgDraw.draw_recs_by_txt(label_path, img, 2, 'black', True)
        img.save(path.join(label_dir, img_name))

# TODO：检查铭牌标签是否出错
if test_crop_img:

    # TODO：注意label
    image_processing.ImageProcess.random_crop(
        img_dir, label_dir, output_dir, 50, 0.4, 0.2, 'number'
    )
    image_processing.ImageProcess.random_crop(img_dir, label_dir, output_dir, 5, 0.4, 0.2, 'plate')

if test_merge_json:

    recdata_io.RecdataIO.merge_json(json1_dir, json2_dir, output_dir, json2_keyword='plate')
    recdata_io.RecdataIO.json_to_txt(output_dir)

if test_correct_one_img:
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