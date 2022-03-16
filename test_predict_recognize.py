# -*- coding: utf-8 -*-
"""
Created on 2022-03-16 20:22:20

@author: Li Zhi
"""
import time

from PIL import Image

from pkgs.east import east_net
from pkgs.recdata import recdata_processing
from pkgs.tool import visualization

EastNet = east_net.EastNet
RecdataRecognize = recdata_processing.RecdataRecognize
RecDraw = visualization.RecDraw
east = EastNet()

def predict_recognize(img):
    recs_list = east.predict(img_dir_or_path=img_filepath)[0]
    recognize_recs_list = RecdataRecognize.recognize(img, 'test.jpg', recs_list)
    RecDraw.draw_recs(recognize_recs_list, img)

    return recognize_recs_list

if __name__ == '__main__':
    start = time.process_time()
    # TODO：矫正有问题
    img_filepath = './video0_0_5_000039.jpg'
    img = Image.open(img_filepath)
    predict_recognize(img)
    img.save('./result.jpg')
    end = time.process_time()
    print(end - start)