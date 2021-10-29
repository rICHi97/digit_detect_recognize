# -*- coding: utf-8 -*-
"""
Created on 2021-10-02 14:39:31

@author: Li Zhi
"""

"""
本模块用于实现可视化，包括端子排图片可视化以及绘图可视化
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import recdata_io
import recdata_processing

def _get_img(img):

    # TODO：路径检查
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, str):
        return Image.open(img)

def _get_font(font, size=20):

    if isinstance(font, (ImageFont.ImageFont, ImageFont.FreeTypeFont, ImageFont.TransposedFont)):
        return font
    elif isinstance(font, str):
        return ImageFont.truetype(font, size)
    elif font is None:
        return None

class ImgDraw(object):
    """
    主要用于在端子排图片上绘制
    """
    def draw_rec(xy_list_or_shape_data, img, width, color, distinguish_first_side=False):
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
            xy_list = recdata_processing.RecData.get_xy_list(xy_list_or_shape_data)
        xy_list = recdata_processing.RecDataProcessing.reorder_rec(xy_list)
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

    def draw_recs(recs_xy_list_or_shape_data, img, width, color, distinguish_first_side=False):
        """
        基于多个端子的四点坐标在图片中绘制端子
        Parameters
        ----------
        recs_xy_list_or_shape_data：多个rec的四点坐标或shape data

        Returns
        ----------
        """
        for xy_list_or_shape_data in recs_xy_list_or_shape_data:  #pylint: disable=E1133
            ImgDraw.draw_rec(xy_list_or_shape_data, img, width, color, distinguish_first_side)

    def draw_recs_by_txt(txt_path, img, width, color, distinguish_first_side=False):
        """
        Parameters
        图片中绘制多个rec端子，基于txt
        ----------
        txt_path：rec txt路径

        Returns
        ----------
        """
        recs_xy_list = recdata_io.RecDataIO.read_rec_txt(txt_path)
        ImgDraw.draw_recs(recs_xy_list, img, width, color, distinguish_first_side)


    def draw_text(text, xy_list, img, color, font=None, precision=2):
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
        center = recdata_processing.RecData.get_center(xy_list)
        draw = ImageDraw.Draw(_get_img(img))
        if isinstance(text, float):
            text = f'{text:.{precision}f}'
        # TODO：考虑不设置font
        draw.text(center, text, fill=color, font=_get_font(font))
        
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
        draw = ImageDraw.Draw(_get_img(img))
        groups = divide_groups['index']
        for group_index, group in enumerate(groups):
        	# TODO：写中文时需传入中文字体
            text = f'group_{group_index + 1}'
            for index_ in group:
                xy_list = recs_xy_list[index_]
                ImgDraw.draw_text(text, xy_list, img, color)
        
class GraphDraw:
    pass