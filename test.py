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

recs_xy_list = recdata_io.RecDataIO.read_rec_txt('2_original.txt')
reg_shape_data = recdata_correcting.Regression.regression(
	recs_xy_list,
	independent=['center'],
	dependent=['length_H', 'length_W', 'rotate_angle_W']
)

end = time.process_time()
print(end - start)