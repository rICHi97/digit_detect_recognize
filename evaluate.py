# -*- coding: utf-8 -*-
"""
Created on 2022-04-09 01:02:17

@author: Li Zhi
"""
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

def evaluate(img_dir, label_dir, predict):
    imgs = os.listdir(img_dir)
    labels = os.listdir(label_dir)
    sigma_m_p, sigma_m_r, cnt_predict_rec, cnt_label_rec, cnt_classes, total_time = 0, 0, 0, 0, 0, 0

    i = 0
    for img_filename in imgs:
        label_filename = img_filename.split('.')[0] + '.txt'
        img_filepath = path.join(img_dir, img_filename)
        label_filepath = path.join(label_dir, label_filename)
        # 预测一张img
        img_recs_list, t = predict(img_dir_or_path=img_filepath, return_time=True)
        img_recs_list = img_recs_list[0]
        total_time += t
        predict_recs_xy_list = []
        label_recs_list = RecdataIO.read_rec_txt(label_filepath)
        label_recs_xy_list = [rec.xy_list for rec in label_recs_list]
        # 预测结果，精确率
        for rec in img_recs_list:
            predict_recs_xy_list.append(rec.xy_list)
            # 和检测结果最匹配的label
            m_max, max_xy_list = most_match(rec.xy_list, label_recs_xy_list)
            sigma_m_p += m_max
            cnt_predict_rec += 1
            if max_xy_list is not None:
                classes = label_recs_list[label_recs_xy_list.index(max_xy_list)].classes
                if rec.classes == classes:
                    cnt_classes += 1
        # 召回结果，召回率
        for rec in label_recs_list:
            # 和label最匹配的检测结果
            m_max, _ = most_match(rec.xy_list, predict_recs_xy_list)
            sigma_m_r += m_max
            cnt_label_rec += 1

        print(i)
        i += 1

    p = sigma_m_p / cnt_predict_rec
    r = sigma_m_r / cnt_label_rec
    f1 = 2 * p * r / (p + r)
    t = total_time / i
    recognition_p = cnt_classes / cnt_predict_rec

    return p, r, f1, t, recognition_p

if __name__ == '__main__':
    # 只能用第一次运行作标准，后续相同数据可能会缓存
    east = EastNet(backdone='vgg', training=False, fine_tune=False, bidirectional=False)
    p, r, f1, t, recognition_p = evaluate('./resource/test_data/image', './resource/test_data/label_txt', east.predict)
