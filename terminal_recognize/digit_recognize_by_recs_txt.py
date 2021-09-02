# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:10:09 2021

@author: LIZHi
"""

# !/usr/bin/python
# -*- coding: utf-8 -*-
"""predict text from images docstrings.

OCR模型(east+crnn)识别图片中的文字, Input:images, Output:text dictionary

    $python predict.py

Version: 0.1
"""
import os

import time
import cv2 as cv
import numpy as np
from keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont

# from PIL import ImageDraw, ImageFont
from math import degrees, atan2, fabs, sin, cos, radians
from data_process.clustering_regression_recs_correction import *
from data_process.clustering_regression_recs_correction import _get_rec_center 
from tools.digit_recognize.digit_recognize_ import *
from tools.results_visualize.draw_on_image import *
from data_process.image_segment import segment_img_to_east
from data_process.image_joint import joint_img_
def resize_image(im, max_img_size=832):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:     # 起到and的作用
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height

    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:                                     
        o_width = im_width
    
    
    # fixme 最多裁剪31个pixel 是否影响边缘效果
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def dumpRotateImage(img, rec):
    
    # xDim, yDim = img.shape[1], img.shape[0]
    # fixme 扩展文字白边 参数为经验值 原始为0.02 0.05
    # NOTICE反转，原始为正
    xlength = int((rec[6] - rec[2]) * 0.02)
    ylength = int((rec[7] - rec[3]) * 0.02)
    rt = [rec[0], rec[1]]
    lt = [rec[2], rec[3]]
    rb = [rec[6], rec[7]]
    lb = [rec[4], rec[5]]
    degree = degrees(atan2( rb[1] - lb[1],  
                            rb[0] - lb[0]))
    # degree = (1.0667 - 0.014 * degree) * degree
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2   # fixme 扩展宽高 否则会被裁剪
    imgRotation = cv.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    # 旋转之后的坐标
    [[rt[0]], [rt[1]]] = np.dot(matRotation, np.array([[rt[0]], [rt[1]], [1]]))    
    [[lt[0]], [lt[1]]] = np.dot(matRotation, np.array([[lt[0]], [lt[1]], [1]]))
    [[rb[0]], [rb[1]]] = np.dot(matRotation, np.array([[rb[0]], [rb[1]], [1]]))
    [[lb[0]], [lb[1]]] = np.dot(matRotation, np.array([[lb[0]], [lb[1]], [1]]))
    xmin, ymin = min([rt[0], lt[0], rb[0], lb[0]]), min([rt[1], lt[1], rb[1], lb[1]])
    xmax, ymax = max([rt[0], lt[0], rb[0], lb[0]]), max([rt[1], lt[1], rb[1], lb[1]])
    ydim, xdim = imgRotation.shape[:2]
    xlen = int((xmax - xmin) * 0.03) * 0
    ylen = int((ymax - ymin) * 0.02) * 0
    lt_N = []
    rb_N = []
    lt_N.append(max(1, int(xmin) - xlen))
    lt_N.append(max(1, int(ymin) - ylen))
    rb_N.append(min(xdim - 1, int(xmax) + xlen))
    rb_N.append(min(ydim - 1, int(ymax) + ylen))

    imgRotation = np.uint8(imgRotation)
    img_rot = Image.fromarray(imgRotation)
    
    rec = (lt_N[0], lt_N[1], rb_N[0], rb_N[1])
    img_rec = img_rot.crop(rec)
    return img_rec


# 输入为cv图片
def process_image(img, height = 30, normalize = True, color_inv = False, blur = True, thres = True, morphology = True):
    scale = img.size[1] * 1.0 / height
    if scale == 0:
        return
    w = int(img.size[0] / scale)
    if w == 0:
        return
    img = img.resize((w, height), Image.ANTIALIAS)
    img_in = np.array(img)
    img_out = np.zeros(img_in.shape, np.uint8)

    if normalize:
        cv.normalize(img_in, img_out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    else:
        img_out = img_in    
    if blur:
        img_out = cv.GaussianBlur(img_out, (3, 3), 0)
    if color_inv:
        img_out = 255 - img_out  
    if thres:
        ret, img_out = cv.threshold(img_out, 0, 255, cv.THRESH_TOZERO + cv.THRESH_OTSU)
        # _, img_out = cv.threshold(img_out, int(0.9 * ret), 255, cv.THRESH_BINARY)
    if morphology:
        morph_shape = cv.MORPH_RECT
        kernel = cv.getStructuringElement(morph_shape, (2, 1))
        img_out = cv.dilate(img_out, kernel, iterations = 1) 

    digit_area_img = Image.fromarray(img_out.astype(np.int32))
    digit_area_img = digit_area_img.resize((w, height), Image.ANTIALIAS).convert('L')
    return digit_area_img

def take_index(elem):
     return int(elem.split('.')[0][10:])
 
image_dir = "./image/"
# 存放east检测结果txt
recs_dir = './results/east_detect/image_txt/'
digit_dir = './results/digit_recognize/'
imgs_all = os.listdir(image_dir)
draw_font = ImageFont.truetype(font = 'simhei.ttf', size = 20)
font_width, font_height = draw_font.getsize('1')[0], draw_font.getsize('1')[1]
for img_file in imgs_all:
    start = time.clock()
    img = Image.open(image_dir + img_file)
    draw = ImageDraw.Draw(img)
    width, height = img.width, img.height
    # 不包括拓展名
    img_name = img_file.split('.')[0]
    if img_name != 'terminal_29':
        continue
    txt_file = img_name + '_original.txt'
    recs_all = []
    with open(recs_dir + txt_file) as recs_txt:
        lines = recs_txt.readlines()
    for line in lines:
        rec = [float(x) for x in line.split(',')]
        # recs_all之前没有清空
        recs_all.append(rec)

    new_recs_all, rec_index_group_list = recs_correction(recs_all, width, height)
    
    work_dir = os.getcwd()
    for i in range(len(rec_index_group_list)):
        rec_index_group = rec_index_group_list[i]
        rec_group = [new_recs_all[x] for x in rec_index_group]
        digit_areas = get_all_digit_areas(rec_group)
   
        os.chdir(digit_dir)
        if not os.path.exists(img_name + '/group%d'%(i)):
            os.makedirs(img_name + '/group%d'%(i))
        if not os.path.exists(img_name + '/joint_img/group%d'%(i)):
            os.makedirs(img_name + '/joint_img/group%d'%(i))
        with open(img_name + '_group%d.txt'%(i), 'w') as digit_txt:
            for digit_area in digit_areas:
                line = ','.join(['%.2f'%(x) for x in digit_area])
                line += '\n'
                digit_txt.write(line)

        j = 0
        for digit_area in digit_areas:
            _ = image.img_to_array(img)
            digit_area_img = dumpRotateImage(_, digit_area).convert('L')
            digit_area_img = process_image(digit_area_img)
            digit_area_img.save(img_name+ '/group%d/digit%d.jpg'%(i, j))
            j += 1

        sequential_digit_position = joint_img_(img_name + '/group%d/'%(i), img_name + '/joint_img/group%d/'%(i))
        joint_img_dir = img_name + '/joint_img/group%d/'%(i)
        joint_imgs_all = os.listdir(joint_img_dir)

        joint_imgs_all.sort(key = take_index)
        digit_index = 0
        previous_digit_cnt = 0
        start_number = 0
        for joint_img in joint_imgs_all:
            joint_img_name = joint_img.split('.')[0]
            this_sequential_digit_position = sequential_digit_position[joint_img_name]
            recognize_digit_and_position = digit_recognize(joint_img_dir + joint_img)
            if recognize_digit_and_position == None:
                previous_digit_cnt = len(this_sequential_digit_position) - 1
                continue
            start_number = get_start_number(this_sequential_digit_position, recognize_digit_and_position, previous_digit_cnt)
            previous_digit_cnt = len(this_sequential_digit_position) - 1

        for digit_index in range(len(rec_group)):
            rec = rec_group[digit_index]
            center = _get_rec_center(rec)
            draw.text(center, '%d'%(digit_index + start_number), fill = 'black', font = draw_font)
            draw.line([tuple([rec[0], rec[1]]),
                       tuple([rec[2], rec[3]]),
                       tuple([rec[4], rec[5]]),
                       tuple([rec[6], rec[7]]),
                       tuple([rec[0], rec[1]])], width = 2, fill = 'blue')
        os.chdir(work_dir)
    img.save(digit_dir + '%s.jpg'%(img_name))
    end = time.clock()
    print(end - start
          )