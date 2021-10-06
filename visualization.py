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

	if isinstance(
		font, 
		(ImageFont.ImageFont, ImageFont.FreeTypeFont, ImageFont.TransposedFont)
	):
		return font
	elif isinstance(font, str):
		return ImageFont.truetype(font, size)

class ImgDraw(object):

	def draw_rec(xy_list, img, width, color, distinguish_first_side=False):
		"""
		图片中绘制rec端子
		Parameters
		----------
		xy_list：rec的四点坐标
		img：PIL的Image object，端子对应图片
		color：rec线颜色
		distinguish：是否区分首边

		Returns
		----------
		"""
		draw = ImageDraw.Draw(_get_img(img))
		xy_list = recdata_processing.RecDataProcessing.reorder_rec(xy_list)
		xy_list = np.reshape((4, 2)).tolist()
		last_edge = (xy_list[3], xy_list[0])
		draw.line(xy_list, color, width)
		draw.line(last_edge, color, width)

		if distinguish_first_side:
			first_edge = (xy_list[0], xy_list[1])
			if color == 'yellow':
				color = 'blue'
			else:
				color = 'yellow'
			draw_line(first_edge, width, color)		

	def draw_recs_txt(txt_path, img, width, color, distinguish_first_side=False):
		"""
		Parameters
		图片中绘制多个rec端子，基于txt
		----------
		txt_path：rec txt路径

		Returns
		----------
		"""
		recs_xy_list = recdata_io.RecDataIO.read_rec_txt(txt_path)
		for xy_list in recs_xy_list:
			RecDataDraw.draw_rec(xy_list, img, width, color, distinguish_first_side)

	def draw_rec_text(text, xy_list, img, color, font=None, precision=2):
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
		draw.text(center, text, fill=color, font=_get_font(font))
		

class GraphDraw:
	pass