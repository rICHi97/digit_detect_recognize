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
test_reg_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)

end = time.process_time()
print(end - start)