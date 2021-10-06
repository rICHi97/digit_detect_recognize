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

start = time.process_time()
recs_xy_list = recdata_io.RecDataIO.read_rec_txt('1_original.txt')
pca_values = recdata_correcting.PCA_.get_pca_values(recs_xy_list)
delta_values = recdata_correcting.PCA_.get_delta_values(1, pca_values)
img = Image.open('1.jpg')
for xy_list, delta_value in zip(recs_xy_list, delta_values):
	visualization.ImgDraw.draw_rec_text(delta_value, xy_list, img, 'black', 'arial.ttf')

# for xy_list, pca_value in zip(recs_xy_list, pca_values):
	# visualization.ImgDraw.draw_rec_text(pca_value, xy_list, img, 'black', 'arial.ttf')
img.save('1_test.jpg')
end = time.process_time()
print(end - start)
