# -*- coding: utf-8 -*-
"""
Created on 2022-04-11 22:58:16

@author: Li Zhi
"""
import random
import os
from os import path

import numpy as np
from shapely import geometry

from pkgs.east import east_net
from pkgs.recdata import recdata_io

Polygon = geometry.Polygon
EastNet = east_net.EastNet
RecdataIO = recdata_io.RecdataIO

def most_match(xy_list, xy_list_set):
    polygon = lambda xy_list: Polygon(np.array(xy_list).reshape((4, 2)))
    r1 = polygon(xy_list)
    m_max = 0
    max_xy_list = None
    for _ in xy_list_set:
        r2 = polygon(_)
        i = r1.intersection(r2).area
        u = r1.area + r2.area
        m = 2 * i / u
        if m > m_max:
            m_max = m
            max_xy_list = _
    return m_max, max_xy_list

def inspect_defect(img_dir, label_dir, predict, cnt):
    defect = {}
    imgs = os.listdir(img_dir)
    imgs = [img for img in imgs if 'terminal_1_' in img or 'terminal_2_'  in img or 'terminal_3_' in img or 'video0_0_' in img]
    labels = os.listdir(label_dir)
    i = 0
    while i <= cnt:
        sigma_m_p = 0
        img_filename = random.choice(imgs)
        label_filename = img_filename.split('.')[0] + '.txt'
        img_filepath = path.join(img_dir, img_filename)
        label_filepath = path.join(label_dir, label_filename)
        # 预测一张img
        img_recs_list, _ = predict(img_dir_or_path=img_filepath, return_time=True)
        img_recs_list = img_recs_list[0]
        if len(img_recs_list) == 0:
            continue
        label_recs_list = RecdataIO.read_rec_txt(label_filepath)
        label_recs_xy_list = [rec.xy_list for rec in label_recs_list]
        # 预测结果，精确率
        for rec in img_recs_list:
            # 和检测结果最匹配的label
            m_max, max_xy_list = most_match(rec.xy_list, label_recs_xy_list)
            sigma_m_p += m_max
        i += 1

        p = sigma_m_p / len(img_recs_list)
        defect[img_filename] = float(f'{p:.4f}')

    return defect

if __name__ == '__main__':
    east = EastNet(backdone='vgg', training=False, fine_tune=False, bidirectional='V2')
    defect = inspect_defect('./resource/train_data/a_img_1/', './resource/train_data/a_txt_1/', east.predict, 58)
    lines = [f'{key}:{value}\n' for key, value in defect.items()]
    with open('defect.txt', 'w') as t:
        t.writelines(lines)
        