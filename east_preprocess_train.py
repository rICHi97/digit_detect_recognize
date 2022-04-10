# -*- coding: utf-8 -*-
"""
Created on 2021-12-29 01:34:54

@author: Li Zhi
"""
from PIL import Image
from tensorflow import keras

from pkgs.east import east_data, east_net, network
from pkgs.recdata import recdata_io
from pkgs.tool import visualization

EastPreprocess = east_data.EastPreprocess
EastNet = east_net.EastNet
RecdataIO = recdata_io.RecdataIO
RecDraw = visualization.RecDraw

# python east_preprocess_train.py; sleep 300; sh /mistgpu/shutdown.sh
if __name__ == '__main__':
    # EastPreprocess.preprocess()
    # EastPreprocess.label()
    east = EastNet(backdone='vgg', training=True, fine_tune=True, bidirectional=True)
    # east = EastNet(backdone='pva', training=False, fine_tune=False)
    # east.train()
    # terminal_5_terminal_3, 30, 39
    # img_path = './resource/test_data/image/video0_0_5_000039.jpg'
    # label_path = './resource/test_data/label_txt/video0_0_5_000039.txt'

    # r_l = east.predict(img_dir_or_path=img_path)
    # img = Image.open(img_path)
    # RecDraw.draw_recs(r_l[0], img)
    # img.show()
    # east.plot()

    # img2 = Image.open(img_path)
    # label_recs = RecdataIO.read_rec_txt(label_path)
    # RecDraw().draw_recs(label_recs, img2)
    # img2.show()