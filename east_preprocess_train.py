# -*- coding: utf-8 -*-
"""
Created on 2021-12-29 01:34:54

@author: Li Zhi
"""
from PIL import Image
# from tensorflow import keras

# from pkgs.east import east_data, east_net, network
from pkgs.recdata import recdata_io, recdata_correcting
from pkgs.tool import visualization

# EastPreprocess = east_data.EastPreprocess
# EastNet = east_net.EastNet
RecdataIO = recdata_io.RecdataIO
RecDraw = visualization.RecDraw
Correction = recdata_correcting.Correction

# python east_preprocess_train.py; sleep 300; sh /mistgpu/shutdown.sh
if __name__ == '__main__':
    # EastPreprocess.preprocess()
    # EastPreprocess.label()
    # east = EastNet(backdone='vgg', training=False, fine_tune=False, bidirectional='V2')
    # east = EastNet(backdone='pva', training=False, fine_tune=False)
    # east.train()
    # terminal_5_terminal_3, 30, 39
    # img_path = './resource/train_data/a_img_1/terminal_3_terminal_14.jpg'
    img_path = './video0_0_5_000039_plate_3.jpg'
    # label_path = './video0_0_5_000039_plate_3.txt'
    label_path = './predict_video0_0_5_000039_plate_3.txt'
    # label_path = './terminal_3_terminal_14.txt'

    # r_l = east.predict(img_dir_or_path=img_path)
    # img = Image.open(img_path)
    # c_r_l = Correction().correct_rec(r_l[0])
    # RecDraw.draw_recs(c_r_l, img)
    # RecDraw.draw_recs(r_l[0], img)
    # img.show()

    img2 = Image.open(img_path)
    label_recs = RecdataIO.read_rec_txt(label_path)
    c_label_recs = Correction().correct_rec(label_recs)
    RecDraw().draw_recs(c_label_recs, img2)
    # RecDraw().draw_recs(label_recs, img2)
    img2.show()