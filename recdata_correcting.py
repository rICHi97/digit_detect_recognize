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
import visualization

LinearRegression = linear_model.LinearRegression
PCA_ = decomposition.PCA
EPSILON = 1e-4

_filter_nan_data = lambda data_list: [data for data in data_list if not math.isnan(data)]

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
    avg,
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

    def extract_dict_to_list(data_dict, avg, dest_list, key):

        # 中心、长度、倾斜角各有两维
        # 对中心xy坐标取平均似乎没有意义
        if avg:
            dest_list.append(np.array(data_dict[key]).mean())
        else:
            dest_list.append(data_dict[key][0])
            dest_list.append(data_dict[key][1])

        return dest_list

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
        keys = []
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
            for i, this_coef in enumerate(coef):
                data_list[i] *= this_coef

        # 已获取1 * n_features数据
        sklearn_data.append(data_list)

    sklearn_data = np.array(sklearn_data)

    return sklearn_data
     

# TODO：只有前两个delta_pca_value可能是nan
def _get_avg(data_list):
    """
    求平均值，不考虑nan
    """
    data_list = _filter_nan_data(data_list)
    avg = sum(data_list) / len(data_list)

    return avg

def _get_filter_data(data_list, filter_nan=True):
    """
    通过标准差过滤数据
    """
    # TODO：注意元素长度可能为0
    filter_data, filter_threshold = [], 2.5
    nonan_data_list = _filter_nan_data(data_list)
    avg = _get_avg(nonan_data_list)
    std = np.sqrt(np.mean((np.array(nonan_data_list) - avg) ** 2))
    for data in data_list:
        # (是nan and filter_nan = False) or ((data - avg) < filter_threshold * std)的数据保留
        if not filter_nan and math.isnan(data):
            filter_data.append(data)
        # 'nan' -a > b is False, 'nan' - a < b is False
        elif abs(data - avg) < filter_threshold * std:
            filter_data.append(data)

    return filter_data

# TODO：同一张图片的data保存以便后续使用
# tarin_data_dict = {'img_name':, 'train_data':}

class PCA(object):
    """
    PCA及基于pca value的分组
    """
    # TODO：一定需要归一化？
    def get_pca_values(  #pylint: disable=E0213
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
        return_instance=False,
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
        pca_ = PCA_(n_components=1).fit(sklearn_data)
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

    def divide_recs(pca_values):
        """
        通过PCA值对端子分组，端子分布分为三类：单列、双列、单列多线
        当avg_delta_two_value大致为avg_delta_one_value两倍时，为单列
        否则为双列，双列时判断pca_value正负号
        单列时，逐个判断delta_one_value与avg_delta_one_value
        Parameters
        ----------
        recs_xy_list：多个rec的四点坐标

        Returns
        ----------
        divide_groups：分组字典
        --'index'：list，元素为每组的index
        --'value'：list，每组的pca_value
        """
        def get_delta_ratio(pca_values):

            delta_one_values, delta_two_values = (
                PCA.get_delta_values(1, pca_values),
                PCA.get_delta_values(2, pca_values)
            )
            avg_delta_one_value, avg_delta_two_value = (
                _get_avg(delta_one_values),
                _get_avg(delta_two_values)
            )
            delta_ratio = abs(avg_delta_two_value) / abs(avg_delta_one_value + EPSILON)

            return delta_ratio

        def divide_one_col(pca_values):

            delta_one_values = PCA.get_delta_values(1, pca_values)
            avg_delta_value = _get_avg(_get_filter_data(delta_one_values))
            # 逐个比较
            cnt_leap, threshold_leap = 0, 1.6
            divide_groups = {'index': [], 'value': []}
            tmp_group_index, tmp_group_value = [], []

            for index_, delta_value in enumerate(delta_one_values):
                if(
                    math.isnan(delta_value)
                    or abs(delta_value) < threshold_leap * abs(avg_delta_value)
                ):
                    tmp_group_index.append(index_)
                    tmp_group_value.append(pca_values[index_])
                # 此时认定发生跳跃，是一列中的另一条线
                else:
                    divide_groups['index'].append(tmp_group_index)
                    divide_groups['value'].append(tmp_group_value)
                    cnt_leap += 1
                    tmp_group_index.clear()
                    tmp_group_value.clear()

            divide_groups['index'].append(tmp_group_index)
            divide_groups['value'].append(tmp_group_value)

            return divide_groups

        def divide_two_cols(pca_values):

            group1_index, group2_index = [], []
            group1_value, group2_value = [], []
            for index_, value in enumerate(pca_values):
                if value > 0:
                    group1_index.append(index_)
                    group1_value.append(value)
                else:
                    group2_index.append(index_)
                    group2_value.append(value)
            divide_groups = {
                'index': [group1_index, group2_index],
                'value': [group1_value, group2_value]
            }

            return divide_groups

        # TODO：命名还需考虑
        # TODO：枚举
        distribution_types = ('one_col', 'two_cols', 'one_col_n_lines')
        # 端子如果单列分布，avg_delta_two_value大致为avg_delta_one_value的两倍
        # 考虑一定的裕度
        delta_ratio = get_delta_ratio(pca_values)
        one_col_range = [1.8, 2.5]
        is_num_in_range = (
            lambda num, range_: True if num > range_[0] and num < range_[1] else False
        )

        # 此时为单列分布
        if is_num_in_range(delta_ratio, one_col_range):
            divide_groups = divide_one_col(pca_values)
        # 此时为双列分布
        else:
            distribution_type = distribution_types[1] # 'two_cols'
            divide_groups = divide_two_cols(pca_values)

        return divide_groups


class Regression(object):

    def _correct_rec_center(center_xy_data):
        """
        在对端子矫正前，先对xy坐标进行预矫正，基本思路是通过y坐标矫正x坐标
        Parameters
        ----------
        center_xy_data：多个端子的中心坐标，n_samples * n_features形状

        Returns
        corrected_xy_data：矫正后xy中心坐标
        ----------
        """
        # 回归确定整体走向
        center_x_data, center_y_data = center_xy_data[:, 0], center_xy_data[:, 1]
        delta_x, delta_y = (
            np.max(center_x_data) - np.min(center_x_data),
            np.max(center_y_data) - np.min(center_y_data),
        )
        reg = LinearRegression().fit(center_x_data.reshape(-1, 1), center_y_data)
        k = 1 if reg.coef_[0] > 0 else -1

        # 认定y坐标跨度大，通过y分布矫正x分布
        new_center_x_data, new_center_y_data = [], []
        for i in range(len(center_x_data)):
            ratio = (center_y_data[i] - np.min(center_y_data)) / delta_y
            new_center_x = k * ratio * delta_x + np.min(center_x_data)
            new_center_y = k * ratio * delta_y + np.min(center_y_data)
            new_center_x_data.append(new_center_x)
            new_center_y_data.append(new_center_y)

        new_center_xy_data = np.array([new_center_x_data, new_center_y_data]).T

        return new_center_xy_data

    # TODO：比较归一化对结果的影响
    def regression(
        recs_xy_list,
        norm_width=None,
        norm_height=None,
        coef=None,
        avg=False,  # TODO：暂时不考虑avg，默认所有变量都不取平均
        return_original=False,
        **regression_vars,
        ):
        """
        通过回归矫正端子
        回归自变量暂定中心坐标
        回归因变量暂定W边的倾斜角，H、W边的长度
        Parameters
        ----------
        recs_xy_list：多个rec的四点坐标
        norm_width：归一化宽度
        norm_height：归一化高度
        coef：数据回归前乘比例系数
        avg：length、rotate_angle是否取两条对边的平均值
        regression_vars：指定回归变量的字典，格式为
        --'independent': ['center']
        --'dependent': ['length_H', 'length_W', 'rotate_angle_W']

        Returns
        regression_rec_shape_data：rec data字典
        区别于rec_shape_data，仅包括回归自变量以及回归因变量
        ----------
        """
        def get_regression_data(regression_vars):

            vars_ = {
                'center': False,
                'length_W': False,
                'length_H': False,
                'rotate_angle_W': False,
                'rotate_angle_H': False,
            }

            for key in vars_.keys():
                if key in regression_vars:
                    vars_[key] = True

            regression_data = _get_train_data(
                recs_xy_list,
                norm_width,
                norm_height,
                vars_['center'],
                vars_['length_W'],
                vars_['length_H'],
                vars_['rotate_angle_W'],
                vars_['rotate_angle_H'],
                coef,
                avg,
            )
            original_regression_data = regression_data.copy()
            # 修正xy数据
            if 'center' in regression_vars:
                if avg:
                    center_index = regression_vars.index('center')
                else:
                    center_index = 2 * regression_vars.index('center')
                center_xy_data = regression_data[:, center_index:center_index + 2]
                new_center_xy_data = Regression._correct_rec_center(center_xy_data)
                regression_data[:, center_index:center_index + 2] = new_center_xy_data

            return regression_data, original_regression_data

        independent_vars, dependent_vars = (
            regression_vars['independent'], regression_vars['dependent']
        )
        independent_data, dependent_data = (
            get_regression_data(independent_vars), get_regression_data(dependent_vars)
        )
        reg = LinearRegression().fit(independent_data[0], dependent_data[0])

        def extract_array_to_dict(dest_dict, train_data, vars_):
            for var in vars_:
                # 每个变量有两维，对应对边
                start, end = 2 * vars_.index(var), 2 * vars_.index(var) + 2
                # 认定array格式为n_samples * n_features
                var_data = train_data[:, start:end]
                dest_dict[var] = var_data.tolist()

        regression_dependent_data = reg.predict(independent_data[0])
        regression_rec_shape_data = {}
        extract_array_to_dict(
            regression_rec_shape_data, independent_data[0], independent_vars,
        )
        extract_array_to_dict(
            regression_rec_shape_data, regression_dependent_data, dependent_vars,
        )

        return regression_rec_shape_data


class Correction(object):

    min_rec_cnt_to_correct = 3
    reg_indep_vars = ['center']
    reg_dep_vars = ['length_W', 'length_H', 'rotate_angle_W']

    # TODO：求平均倾斜角
    def _get_avg_rotate_angle_H(recs_xy_list):

        rotate_angle_H1_list, rotate_angle_H2_list = [], []
        for xy_list in recs_xy_list:  #pylint: disable=E1133
            rotate_angle_H_dict = recdata_processing.RecData.get_rec_shape_data(
                xy_list,
                center=False,
                length_W=False,
                length_H=False,
                rotate_angle_W=False,
                rotate_angle_H=True,
            )
            _ = rotate_angle_H_dict['rotate_angle_H']
            rotate_angle_H1_list.append(_[0])
            rotate_angle_H2_list.append(_[1])

        def get_signed_avg(data_list):

            positive_data = [data for data in data_list if data >= 0]
            negative_data = [data for data in data_list if data < 0]
            if len(positive_data) >= len(negative_data):
                signed_data = positive_data
            else:
                signed_data = negative_data

            signed_avg = _get_avg(_get_filter_data(signed_data))

            return signed_avg

        avg_rotate_angle_H1, avg_rotate_angle_H2 = (
            get_signed_avg(rotate_angle_H1_list),
            get_signed_avg(rotate_angle_H2_list),
        )

        return avg_rotate_angle_H1, avg_rotate_angle_H2

    # TODO：矫正端子，返回矫正后端子数据
    # TODO：矫正前也许需要先LOF筛选异常值
    def correct_rec(recs_xy_list):
        """
        对端子数据进行矫正。
        流程：
        --1.求pca values
        --2.依据pca values分组
        --3.对每组端子回归，回归返回shape data
        --4.补充H方向倾斜角数据得到最终shape data
        Parameters
        ----------
        recs_xy_list：多个rec的四点坐标

        Returns
        ----------
        """
        corrected_recs_shape_data = [None for i in range(len(recs_xy_list))]
        pca_values = PCA.get_pca_values(recs_xy_list)
        divide_groups = PCA.divide_recs(pca_values)
        index_groups = divide_groups['index']

        for index_group in index_groups:
            # 如果个数少于3个，保持原始数据
            # 否则，通过回归矫正
            if len(index_group) < Correction.min_rec_cnt_to_correct:
                for index_ in index_group:
                    xy_list = recs_xy_list[index_]  #pylint: disable=E1136
                    shape_data = recdata_processing.RecData.get_rec_shape_data(xy_list)
                    corrected_recs_shape_data[index_] = shape_data
            else:
                recs_group = [recs_xy_list[i] for i in index_group]  #pylint: disable=E1136
                avg_rotate_angle_H1, avg_rotate_angle_H2 = (
                    Correction._get_avg_rotate_angle_H(recs_group)
                )
                reg_shape_data = Regression.regression(
                    recs_group,
                    norm_width=None,
                    norm_height=None,
                    coef=None,
                    avg=False, 
                    return_original=False,
                    independent=Correction.reg_indep_vars,
                    dependent=Correction.reg_dep_vars,
                )
                for index_ in index_group:
                    # index_group存储每组端子的序号
                    # index_index为序号在group中的位置
                    index_index = index_group.index(index_) 
                    shape_data = {}
                    for key in Correction.reg_dep_vars + Correction.reg_indep_vars:
                        shape_data[key] = reg_shape_data[key][index_index]
                    shape_data['rotate_angle_H'] = [avg_rotate_angle_H1, avg_rotate_angle_H2]
                    corrected_recs_shape_data[index_] = shape_data
        
        return corrected_recs_shape_data
