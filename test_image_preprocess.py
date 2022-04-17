# -*- coding: utf-8 -*-
"""
Created on 2022-04-15 23:16:21

@author: Li Zhi
"""
import cv2
import numpy as np
from PIL import Image

from pkgs.east import east_net
from pkgs.recdata import  recdata_processing, recdata_io
from pkgs.tool import image_processing

EastNet = east_net.EastNet
Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess
RecdataIO = recdata_io.RecdataIO
ImageProcess = image_processing.ImageProcess

if __name__ == '__main__':
    # label_filepath = './video0_0_5_000039_plate_3_原始.txt'
    # img_filepath = './video0_0_5_000039_plate_3.jpg'
    img_filepath = './video0_0_5_000017_terminal_15.jpg'
    east = EastNet(backdone='vgg', training=False, fine_tune=False, bidirectional='V2')
    r_l = east.predict(img_dir_or_path=img_filepath)
    # recs_list = RecdataIO.read_rec_txt(label_filepath)
    # recs_list = RecdataProcess.reorder_recs(recs_list)
    img = Image.open(img_filepath)
    joint_data = ImageProcess.joint_rec(img, 'video0_0_5_000017_terminal_15.jpg', r_l[0])
    # rec = crop_rec(img, recs_list[0].xy_list)
    # rec.show()
    # for i, rec in enumerate(recs_list):
    #     xy_list = rec.xy_list
    #     rec = crop_rec(img, xy_list)
    #     rec = preprocess_img(rec, True)
    #     rec.save(f'./resource/{i}.jpg')
