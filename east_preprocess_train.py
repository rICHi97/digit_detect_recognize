# -*- coding: utf-8 -*-
"""
Created on 2021-12-29 01:34:54

@author: Li Zhi
"""
from PIL import Image
from tensorflow import keras

from pkgs.east import east_data, east_net, network
from pkgs.tool import visualization

EastPreprocess = east_data.EastPreprocess
EastNet = east_net.EastNet
RecDraw = visualization.RecDraw

# python east_preprocess_train.py; sleep 300; sh /mistgpu/shutdown.sh
if __name__ == '__main__':
    # EastPreprocess.preprocess()
    # EastPreprocess.label()
    east = EastNet(backdone='inception_res', training=True, fine_tune=False)
    # east = EastNet(backdone='pva', training=False, fine_tune=False)
    # east.train()

    # r_l = east.predict()
    # img = Image.open('./resource/test_data/image/terminal_3_terminal_20.jpg')
    # RecDraw.draw_recs(r_l[0], img)
    # img.show()
