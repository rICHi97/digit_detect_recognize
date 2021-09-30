# -*- coding: utf-8 -*-
"""
Created on 2021-09-29 20:08:09

@author: Li Zhi
"""

"""
本模块用以实现数据的输入输出，包括从txt文件读取rec数据，将rec数据保存到txt等
一个rec的四个端点：xy_list
一张图片的多个rec的端点：recs_xy_list
多张图片的recs字典：imgs_rec_dict
"""


class RecDataIO(object):
	"""
	读取txt文件，写入txt文件
	"""
	@staticmethod
	def read_rec_txt(txt_path):
		"""
		读取一张图片的rec txt文件
		将其转为该图片中所有rec四点坐标的列表
		Parameters
		----------
		txt_path：rec txt路径

		Returns
		recs_xy_list：多个rec的四点坐标
		----------
		"""
		recs_xy_list = []
		with open(txt_path, 'r', encoding='utf8') as rec_txt:
			lines = rec_txt.readlines().splitlines()
			for line in lines:
				line = line.split(',')
				xy_list = [float(xy) for xy in line]
				recs_xy_list.append(xy_list)

		return recs_xy_list

	@staticmethod
	def read_rec_txt_dir(txt_dir, keyword=None):
		"""
		对一个文件夹中的txt进行read_rec_txt操作
		Parameters
		----------
		txt_dir：rec txt的文件夹
		keyword：如果不为None，则只读取文件名包含keyword的txt
		
		Returns
		----------
		imgs_rec_dict：dict，键为txt文件名，值为该txt的recs_xy_list
		"""
		
		

