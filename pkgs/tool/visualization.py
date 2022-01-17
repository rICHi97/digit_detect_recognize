# -*- coding: utf-8 -*-
"""
Created on 2021-10-02 14:39:31

@author: Li Zhi
本模块用于实现可视化，包括端子排图片可视化以及绘图可视化
"""
import os
from os import path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import cfg
from ..east import east_data
from ..recdata import recdata_io, recdata_processing

EastData = east_data.EastData
EastPreprocess = east_data.EastPreprocess
RecdataIO = recdata_io.RecdataIO


# TODO：读取一个文件夹中的txt文件，在对应的图片上绘制需要进一步封装
def _get_img(img):

    # TODO：路径检查
    if isinstance(img, Image.Image):
        pass
    elif isinstance(img, str):
        img = Image.open(img)

    return img

def _get_font(font, size=32):

    if isinstance(font, (ImageFont.ImageFont, ImageFont.FreeTypeFont, ImageFont.TransposedFont)):
        return font
    elif isinstance(font, str):
        return ImageFont.truetype(font, size)
    elif font is None:
        return ImageFont.truetype('../resource/font/HGBTS_CNKI.TTF')

# TODO：统一输入格式为PIL.Image
# TODO：cfg参数化
# 输入Image，对img即时修改，方法外手动保存
# 输入路径，那就得在api内部保存图片，参数需要增加输出路径
# 内部保存图片不利于对一张图片调用多个draw方法
class RecDraw(object):
    """
    主要用于在端子排图片上绘制
    """
    @staticmethod
    def draw_rec(xy_list_or_shape_data, img, width=2, color='black', distinguish_first_side=False):
        """
        图片中绘制rec端子
        Parameters
        ----------
        xy_list_or_shape_data：rec的四点坐标或shape data
        img：PIL的Image object，端子对应图片
        with：线宽
        color：rec线颜色
        distinguish：是否区分首边

        Returns
        ----------
        """
        draw = ImageDraw.Draw(_get_img(img))
        if type(xy_list_or_shape_data) is list:
            xy_list = xy_list_or_shape_data
        elif type(xy_list_or_shape_data) is dict:
            xy_list = recdata_processing.Recdata.get_xy_list(xy_list_or_shape_data)
        xy_list = recdata_processing.RecdataProcess.reorder_rec(xy_list)
        xy_list = np.reshape(xy_list, (4, 2)).tolist()
        last_edge = (xy_list[3], xy_list[0])
        to_tuple_element = lambda list_: [tuple(element) for element in list_] 
        draw.line(to_tuple_element(xy_list), color, width)
        draw.line(to_tuple_element(last_edge), color, width)

        if distinguish_first_side:
            first_edge = (xy_list[0], xy_list[1])
            if color == 'yellow':
                color = 'blue'
            else:
                color = 'yellow'
            draw.line(to_tuple_element(first_edge), color, width)

    @staticmethod
    def draw_recs(
        recs_xy_list_or_shape_data, img, width=2, color='black', distinguish_first_side=False
    ):
        """
        基于多个端子的四点坐标在图片中绘制端子
        Parameters
        ----------
        recs_xy_list_or_shape_data：多个rec的四点坐标或shape data

        Returns
        ----------
        """
        for xy_list_or_shape_data in recs_xy_list_or_shape_data:  #pylint: disable=E1133
            RecDraw.draw_rec(xy_list_or_shape_data, img, width, color, distinguish_first_side)

    # TODO：绘制整个文件夹中img
    @staticmethod
    def draw_recs_by_txt(txt_path_dir, img_dir, width, color, distinguish_first_side=False):
        """
        Parameters
        图片中绘制多个rec端子，基于txt
        ----------
        txt_path：rec_txt路径或存放txt的dir
        img_dir：PIL Image或存放img的dir

        Returns
        ----------
        """   
        recs_xy_list = recdata_io.RecdataIO.read_rec_txt(txt_path_dir)
        RecDraw.draw_recs(recs_xy_list, img_dir, width, color, distinguish_first_side)

    @staticmethod
    def draw_text(text, xy_list, img, color='black', font=None, precision=2):
        """
        绘制基于rec位置的文本
        Parameters
        ----------
        text：待绘制的文本
        xy_list：文本对应rec的四点坐标
        font：PIL.ImageFont字体

        Returns
        ----------
        """
        center = recdata_processing.Recdata.get_center(xy_list)
        draw = ImageDraw.Draw(_get_img(img))
        if isinstance(text, float):
            text = f'{text:.{precision}f}'
        # TODO：考虑不设置font
        draw.text(center, text, fill=color, font=_get_font(font))
    
    @staticmethod   
    def draw_divide_group(divide_groups, recs_xy_list, img, width, color):
        """
        Parameters
        ----------
        divide_groups：分组数据
        recs_xy_list：多个rec四点坐标
        img：待绘制结果图片
        width：线宽
        color：颜色

        Returns
        ----------
        """
        groups = divide_groups['index']
        for group_index, group in enumerate(groups):
        	# TODO：写中文时需传入中文字体
            text = f'group_{group_index + 1}'
            for index_ in group:
                xy_list = recs_xy_list[index_]
                RecDraw.draw_text(text, xy_list, img, color)
    
    @staticmethod
    def draw_gt(
        gt_path,
        img_path,
        resized=False,
        max_img_size=cfg.gt_max_img_size,
        output_dir=cfg.gt_output_dir,
    ):
        """
        gt文件中的xy_list是基于resize后图片，使用本api前确认img为原始图片还是resize图片
        Parameters
        ----------
        resized：img_path所指img是否为resize后图片。若为原始图片，resized = False。

        Returns
        ----------
        """
        recs_xy_list, recs_classes_list = RecdataIO.read_gt(gt_path)
        d_width, d_height = max_img_size, max_img_size
        img = Image.open(img_path)
        if not resized:
            scale_ratio_w, scale_ratio_h = img.width / d_width, img.height / d_height

        for xy_list, classes in zip(recs_xy_list, recs_classes_list):
            xy_list = np.asarray(xy_list).reshape((4, 2))
            if not resized:
                xy_list[:, 0] *= scale_ratio_w
                xy_list[:, 1] *= scale_ratio_h
            xy_list = xy_list.tolist()
            RecDraw.draw_rec(xy_list, img)
            RecDraw.draw_text(classes, xy_list, img)

        img.save(path.join(output_dir, path.basename(img_path)))


class GraphDraw:
    pass