# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 00:33:27 2020

@author: LIZHi
"""


import os
import shutil

street_image_txt_path = r'D:\Dataset\street_view_house_number\txt'
street_image_path = r'D:\Dataset\street_view_house_number\test'

o_street_txt = os.listdir(street_image_txt_path)
o_street_img = os.listdir(r'D:\GitHub_Project\ocr_chinese-master\train_1000\img')
# for o_txt_name in o_street_txt:
#      o_img_name = o_txt_name[5:-4] + '.png'
#      shutil.copy(os.path.join(street_image_path, o_img_name), r'D:\GitHub_Project\ocr_chinese-master\train_1000\img')

for o_img_name in o_street_img:
    o_new_img_name = 'test_' + o_img_name
    os.rename(os.path.join(r'D:\GitHub_Project\ocr_chinese-master\train_1000\img', o_img_name),
              os.path.join(r'D:\GitHub_Project\ocr_chinese-master\train_1000\img', o_new_img_name)
              )
              