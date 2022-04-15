# -*- coding: utf-8 -*-
"""
Created on 2022-04-15 23:16:21

@author: Li Zhi
"""
import cv2 as cv
import numpy as np
from PIL import Image

from pkgs.recdata import  recdata_processing, recdata_io

Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess
RecdataIO = recdata_io.RecdataIO

# 测试透视变换矫正

def crop_rec(img, xy_list):

    img_array = np.asarray(img, 'f')
    img_w, img_h = img.size
    xy_list = RecdataProcess.reorder_rec(xy_list)
    # 找4组对应点
    src = np.array(xy_list, dtype=np.float32).reshape((4, 2))
    _ = Recdata.get_rec_shape_data(xy_list, False, True, True, False, False)
    w = 0.5 * (_['length_W'][0] + _['length_W'][1])
    h = 0.5 * (_['length_H'][0] + _['length_H'][1])
    vector_W, vector_H = np.array((w, 0)), np.array((0, h))
    left_top = np.array(xy_list).reshape((4, 2))[1]
    right_top = left_top + vector_W
    left_bottom = left_top + vector_H
    right_bottom = left_top + vector_W + vector_H

    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0] = right_top
    dst[1] = left_top
    dst[2] = left_bottom
    dst[3] = right_bottom

    M = cv.getPerspectiveTransform(src, dst)
    img_perspective_array = cv.warpPerspective(
        img_array, M, dsize=(img_w, img_h), borderValue=(0, 0, 0)
    )
    img_perspective = Image.fromarray(np.uint8(img_perspective_array))
    img_perspective.show()
    

if __name__ == '__main__':
    label_filepath = './video0_0_5_000039_plate_3_原始.txt'
    img_filepath = './video0_0_5_000039_plate_3.jpg'
    recs_list = RecdataIO.read_rec_txt(label_filepath)
    xy_list = recs_list[0].xy_list
    crop_rec(Image.open(img_filepath), xy_list)

