# -*- coding: utf-8 -*-
"""
Created on 2021-09-30 22:43:08

@author: Li Zhi
"""

"""
本模块用于实现端子数据的矫正，通过PCA分组，然后对每组数据线性回归
sklearn中各方法输入数据格式为array，形状为n_samples * n_features
样本数 * 特征数
"""
import math

import numpy as np
from sklearn import linear_model, decomposition, preprocessing

import recdata_processing

LinearRegression = linear_model.LinearRegression
PCA = decomposition.PCA
EPSILON = 1e-4

def _get_train_data(
		recs_xy_list,
		norm_width,
		norm_height,
		center,
		length_W,
		length_H,
		rotate_angle_W,
		rotate_angle_H,
		coef,
		avg
		):
		"""
		获取用于sklearn的数据
		Parameters
		----------
		见PCA_参数列表

		Returns
		----------
		sklearn_data：符合sklearn输入要求的数据，n_samples * n_features
		"""	
		sklearn_data = []

		def extract_dict_to_list(data_dict, avg, dist_list, key):
		
			if avg:
				dist_list.append(np.array(data_dict[key]).mean())
			else:
				dist_list.append(data_dict[key][0])
				dist_list.append(data_dict[key][1])

			return dist_list

		for xy_list in recs_xy_list:

			data_list = []
			# 归一化
			xy_list = recdata_processing.RecDataProcessing.from_42_to_18(xy_list)
			if norm_width is not None and norm_height is not None:
				for i in range(4):
					xy_list[2 * i] /= norm_width
					xy_list[2 * i + 1] /= norm_height

			# 获取形状数据
			shape_data = recdata_processing.RecData.get_rec_shape_data(
				xy_list,
				center,
				length_W,
				length_H,
				rotate_angle_W,
				rotate_angle_H
			)

			# 改变数据形状
			if center:
				data_list = extract_dict_to_list(shape_data, False, data_list, 'center')
			keys=[]
			if length_W:
				keys.append('length_W')
			if length_H:
				keys.append('length_H')
			if rotate_angle_W:
				keys.append('rotate_angle_W')
			if rotate_angle_H:
				keys.append('rotate_angle_H')
			for key in keys:
				data_list = extract_dict_to_list(shape_data, avg, data_list, key)

			# 系数
			if coef is not None:
				for i in range(len(coef)):
					data_list[i] *= coef[i]

			# 已获取1 * n_features数据
			sklearn_data.append(data_list)

		sklearn_data = np.array(sklearn_data)

		return sklearn_data


class PCA_(object):
	"""
	PCA及基于pca value的分组
	"""
	def get_pca_values(
		recs_xy_list,
		norm_width=None,
		norm_height=None,
		center=True,
		length_W=False,
		length_H=False,
		rotate_angle_W=False,
		rotate_angle_H=False,
		coef=None,
		avg=True,
		return_instance=False
	):
		"""
		对rec数据降维，目前暂定原始数据为中心坐标，降维至1维
		Parameters
		----------
		recs_xy_list：多个rec的四点坐标
		norm_width：归一化宽度，一般为rec所属图片的宽度
		norm_height：归一化高度，一般为rec所属图片的高度
		center=True：原始数据是否包括中心坐标
		length_W=False：原始数据是否包括宽度向量长度
		length_H=False：原始数据是否包括高度向量长度
		rotate_angle_W=False：原始数据是否包括宽度向量倾斜角
		rotate_angle_H=False：原始数据是否包括高度向量倾斜角
		coef=[1, 1, 1, 1, 1, 1]：原始数据在PCA前的系数，其中center包括xy两个系数
		# TODO：可能需考虑每个特征是否取平均
		avg：length、rotate_angle是否取两条对边的平均值
		return_instance：是否返回PCA实例对象

		Returns
		----------
		pca_values：降维后PCA坐标
		"""
		# TODO：检查coef与True feature是否一致
		sklearn_data = _get_train_data(
			recs_xy_list,
			norm_width,
			norm_height,
			center,
			length_W,
			length_H,
			rotate_angle_W,
			rotate_angle_H,
			coef,
			avg
		)
		pca_ = PCA(n_components=1).fit(sklearn_data)
		pca_values = pca_.transform(sklearn_data)
		pca_values = [value[0] for value in pca_values]
		if not return_instance:
			return pca_values

	def get_delta_values(delta_order, pca_values):
		"""
		由pca value得到差值value
		delta_order = n：
			--i < n: delta_value[i] = float('nan')
			--i >= n:delta_value[i] = pca_value[i] - pca_value[i - n]
		
		Parameters

		----------
		delta_order：差值阶数
		pca_values：降维后pca坐标，默认1维

		Returns
		----------
		delta_values：差值坐标
		"""
		delta_values = []
		for i in range(len(pca_values)):
			if i < delta_order:
				delta_values.append(float('nan'))
			else:
				delta_values.append(pca_values[i] - pca_values[i - delta_order])

		return delta_values

	def _get_avg_delta_value(delta_values):

		delta_values = [
			delta_value for delta_value in delta_values if not math.isnan(delta_value)
		]
		avg_delta_value = sum(delta_values) / len(delta_values)

		return avg_delta_value

	def divide_recs(pca_values):
		"""
		通过PCA值对端子分组，端子分布分为三类：单列、双列、单列多线
		Parameters
		----------
		recs_xy_list：多个rec的四点坐标
		Returns
		----------
		"""
		def get_delta_ratio(pca_values):

			delta_one_values, delta_two_values = (
				PCA_.get_delta_values(1, pca_values), 
				PCA_.get_delta_values(2, pca_values)
			)
			avg_delta_one_value, avg_delta_two_value = (
				PCA_._get_avg_delta_value(delta_one_values),
				PCA_._get_avg_delta_value(delta_two_values)
			)
			delta_ratio = abs(avg_delta_two_value) / abs(avg_delta_one_value + EPSILON)

			return delta_ratio

		# TODO：命名还需考虑
		# TODO：枚举
		distribution_types = ('one_col', 'two_cols', 'one_col_n_lines')
		# 端子如果单列分布，avg_delta_two_value大致为avg_delta_one_value的两倍
		# 考虑一定的裕度
		one_col_range = [1.8, 2.5]
		is_num_in_range = (
			lambda num, range_: True if num > range_[0] and num < range_[1] else False
		)

		# 此时为单列分布
		if is_num_in_range(delta_ratio, one_col_range):

		# 此时为双列分布
		else:
			distribution_type = distribution_types[]
			



