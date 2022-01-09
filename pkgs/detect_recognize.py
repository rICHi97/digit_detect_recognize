# -*- coding: utf-8 -*-
"""
Created on 2021-12-26 15:33:39

@author: Li Zhi
端到端模块，顶层封装
"""
from os import path

from PIL import Image

from .east import east_net
from .recdata import recdata_processing, recdata_io
from .tool import visualization

EastNet = east_net.EastNet
OUTPUT_IMG_PATH = './resource/tmp.jpg'  # 基于根目录运行入口文件
RecdataRecognize = recdata_processing.RecdataRecognize
RecdataIO = recdata_io.RecdataIO
RecDraw = visualization.RecDraw


class EndToEnd(object):
    """
    端到端，实现检测及识别，输出txt结果，绘制结果img
    """
    def __init__(self):
        self.east = EastNet()

    def detect_recognize(self, img_path):
        """
        输入单张图片路径，完成检测及识别
        返回绘制识别结果的图片
        可以输出识别结果到txt，通过EastNet.predict实现
        不设置额外参数，实现细节参数通过各模块cfg文件设置默认值
        Parameters
        ----------
        img_path：图片路径

        Returns
        ----------
        result_img：图片，PIL.Image
        """    
        _ = self.east.predict(img_dir_or_path=img_path)
        recs_xy_list, recs_classes_list = _[0][0], _[1][0]

        img = Image.open(img_path)
        img_name = path.basename(img_path)

        # 不包括识别信息
        # for i, xy_list in enumerate(recs_xy_list):
        #     RecDraw.draw_rec(xy_list, img)
        #     RecDraw.draw_text(recs_classes_list[i], xy_list, img)

        # 区别于上文，这是识别成功的
        recognize_recs_list = RecdataRecognize.recognize(
            img, img_name, recs_xy_list, recs_classes_list
        )
        recs_xy_list, recs_classes_list, recs_text_list = [], [], []
        for rec in recognize_recs_list:
            recs_xy_list.append(rec.xy_list)
            recs_classes_list.append(rec.classes)
            recs_text_list.append(rec.text)
            print(rec.text)
            RecDraw.draw_text(rec.text, rec.xy_list, img)
        RecDraw.draw_recs(recs_xy_list, img)
        img.save(OUTPUT_IMG_PATH)
