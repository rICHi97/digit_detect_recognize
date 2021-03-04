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
import cv2
import numpy as np
from tensorflow.compat.v1 import Session, logging, ConfigProto
from east.net.network import East
from east.predict_east import predict_quad
from keras.preprocessing import image
from PIL import Image

# from PIL import ImageDraw, ImageFont
from math import degrees, atan2, fabs, sin, cos, radians
from data_process.clustering_regression_recs_correction import recs_correction, reorder_rec
from east.data.preprocess import resize_image
from tools import digit_recognize
from data_process.image_segment import segment_img_to_east
from data_process.image_joint import joint_img
# 只显示error
os.environ['TF_MIN_CPP_LOG_LEVEL']='3'
logging.set_verbosity(logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
config=ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth=True
session=Session(config=config)


east_model_weights_file = "./east/model/weights_3T832.031-0.037.h5"
# 首先读取root_image中的图片，对于其中的每张图片，将其分割后输出到segment_image中
# 对segment_image中的图片进行检测，检测完成后清空segment_image
root_image = "./east/test/image/"
root_segment = './east/test/image_segment/'
root_predict = './east/test/predict/'
root_txt = './east/test/image_txt/'

root_rec = './crnn/test/recs'
predict_image_path = './east/test/predict/'
results = './results/'

def dumpRotateImage(img, rec):
    xDim, yDim = img.shape[1], img.shape[0]
    
    # fixme 扩展文字白边 参数为经验值 原始为0.02 0.05
    # NOTICE反转，原始为正
    # xlength = int((rec[4] - rec[0]) * 0.02)
    xlength = -int((rec[4] - rec[0]) * 0.05)
    ylength = int((rec[5] - rec[1]) * 0.05)

    # pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
    pt1 = (max(1, rec[0] + xlength), max(1, rec[1] - ylength))
    pt2 = (rec[6], rec[7])
    # pt3 = (min(rec[4] + xlength, xDim - 2),min(yDim - 2, rec[5] + ylength))
    pt3 = (min(rec[4] - xlength, xDim - 2),min(yDim - 2, rec[5] + ylength))
    degree = degrees(atan2(pt2[1] - pt3[1], pt2[0] - pt3[0]))

    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2   # fixme 扩展宽高 否则会被裁剪
    # imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

    pt1 = list(pt1)
    pt3 = list(pt3)

    # img_rot = Image.fromarray(imgRotation)
    # img_rot.save(root_quad + "xx_rot.jpg")
    
    # 旋转之后的坐标
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    xlen = -int((pt3[0] - pt1[0]) * 0.23)
    ylen = int((pt3[1] - pt1[1]) * 0.02)
    pt1_N = []
    pt3_N = []
    pt1_N.append(max(1, int(pt1[0]) + xlen))
    pt1_N.append(max(1, int(pt1[1]) - ylen))
    pt3_N.append(min(xdim - 1, int(pt3[0]) - xlen))
    pt3_N.append(min(ydim - 1, int(pt3[1]) + ylen))
    

    act_pt2_N = [pt3_N[0], pt1_N[1]]
    act_pt4_N = [pt1_N[0], pt3_N[1]]
    imgRotation = np.uint8(imgRotation)
    img_rot = Image.fromarray(imgRotation)
    
    rec = (act_pt2_N[0], act_pt2_N[1], act_pt4_N[0], act_pt4_N[1])
    img_rec = img_rot.crop(rec)
    return img_rec


if __name__ == '__main__':

    # todo east model predict
    east = East()
    east_model = east.east_network()
    east_model.load_weights(east_model_weights_file)
    
    
    correction_flag = True
    segment_flag = False
    
    # for files in os.listdir(root_image):
    #     # root_image 等于east/test/image
    #     # img_path = os.path.join(root_image, files)
    #     img_path = root_image + '%s'%(files)
    #     im_name = files[:-4]
    #     print("path : %s" % img_path)
    #     print("name : %s" % im_name)
        
    #     if segment_flag:
    #         # 裁切图片
    #         # 图片名.jpg -> 图片名_W%dH%d.jpg -> rec%d_图片名_W%dH%d.jpg
    #         segment_img_to_east(img_path, out_path = root_segment)
        
    for segment_img_file in os.listdir(root_image):
    # for segment_img_file in os.listdir(root_segment):
        im_name = segment_img_file[:-4]
        start = time.clock()
        segment_img_path = (root_image + '%s'%(segment_img_file))
        # segment_img_path = (root_segment + '%s'%(segment_img_file))
        segment_img = image.load_img(segment_img_path).convert('RGB')
        predict_img_width, predict_img_height = resize_image(segment_img)
        scale_ratio_w, scale_ratio_h = segment_img.width / predict_img_width, segment_img.height / predict_img_height
        
        is_terminal_in_img = False
        if not is_terminal_in_img:
            text_recs_all, text_recs_len, img_all = predict_quad(east_model, segment_img, img_name=im_name)
         
        if correction_flag:
            # LOF_data = LOF(text_recs_all)
            # slope = DBSCAN_Clustering(recs_after_LOF)
            recs_corrected = recs_correction(text_recs_all)
            recs_len_corrected = []
            recs_len_corrected.append(len(recs_corrected)) 
            temp_len = len(recs_corrected)
            temp_recs = recs_corrected
        else:
            temp_len = len(text_recs_all)
            temp_recs = text_recs_all
         
        '''
        for i in range(text_recs_len[0]):
            segment_img = image.img_to_array(segment_img)
            img_rec = dumpRotateImage(segment_img, reorder_rec(temp_recs[i])[0]).convert('L')
            scale = img_rec.size[1] * 1.0 / 32
            w = int(img_rec.size[0] / scale)
            img_rec = img_rec.resize((w, 32), Image.BICUBIC,)
            img_in = np.array(img_rec)
            img_out = np.zeros(img_in.shape, np.uint8)
            cv2.normalize(img_in, img_out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            ret, img_out = cv2.threshold(img_out, 118, 255, cv2.THRESH_TOZERO)
            img_rec = Image.fromarray(img_out.astype(np.int32))
            # img_rec.convert('L').save(root_rec + r'/rec%d_%s.jpg'%(i, segment_img_file))
            img_rec.resize((w, 40), Image.BICUBIC).convert('L').save(root_rec + r'/rec%d_%s.jpg'%(i, segment_img_file))
        '''
        
        # 针对无裁切的情况
        # rec坐标经过修正，返回对应于原始图片size的坐标
        with open(root_txt + '%s_corrected.txt'%(im_name), 'w') as corrected_txt:
            for i in range(len(recs_corrected)):
                temp_rec = recs_corrected[i]
                for j in range(4):
                    # 0,2,4,6对应x，1,3,5,7对应y
                    temp_rec[2 * j]     *= scale_ratio_w
                    temp_rec[2 * j + 1] *= scale_ratio_h
                line = ','.join(['%.2f'%(x) for x in temp_rec])
                line += '\n'
                corrected_txt.write(line)
        
        with open(root_txt + '%s_original.txt'%(im_name), 'w') as original_txt:
           for i in range(len(text_recs_all)):
                temp_rec = reorder_rec(text_recs_all[i])[0]
                for j in range(4):
                    # 0,2,4,6对应x，1,3,5,7对应y
                    temp_rec[2 * j]     *= scale_ratio_w
                    temp_rec[2 * j + 1] *= scale_ratio_h
                line = ','.join(['%.2f'%(x) for x in temp_rec])
                line += '\n'
                original_txt.write(line) 
                
        # joint_img(img_path = root_rec, out_path = './crnn/test/recs_joint')
        # os.remove(root_segment + '/%s'%(segment_img_file))
        

        end = time.clock()
        print(end - start)


            

