# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 00:42:39 2020

@author: LIZHi
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from math import sin, cos, atan, atan2, degrees, radians, pi, hypot
from PIL import Image, ImageDraw
epsilon  = 1e-4

def  reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    # 起始点从最小的x开始
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + 1e-4)
                    
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + 1e-4)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    
    return reorder_xy_list

#  from copy import deepcopy
# 得到各边向量
# 返回顺序为W1，W2， H1，H2
# 从起始边逆时针顺序
def _get_side_vector(rec):
    return ((rec[0] - rec[2], rec[1] - rec[3]), 
            (rec[6] - rec[4], rec[7] - rec[5]), 
            (rec[2] - rec[4], rec[3] - rec[5]), 
            (rec[0] - rec[6], rec[1] - rec[7]))

def _unify_length(vector_1, vector_2):
    length_1, length_2 = np.linalg.norm(vector_1), np.linalg.norm(vector_2)
    if length_1 > length_2:
        k = length_1 / length_2
        vector_2 = (k * vector_2[0], k * vector_2[1])
    else:
        k = length_2 / length_1
        vector_1 = (k * vector_1[0], k * vector_1[1])
    return vector_1, vector_2

# 得到rec中心坐标
def _get_rec_center(rec):
    center_x, center_y = 0, 0
    for i in range(4):
        center_x += rec[2 * i] / 4
        center_y += rec[2 * i + 1] / 4
    return center_x, center_y

# 将框四点坐标list转为框中心坐标list
# 输入：框四点坐标list
# 返回：框中心坐标list，框中心x坐标list，框中心y坐标list
def _from_recs_to_centers(recs):
    center_x_list = [] #框中心x坐标
    center_y_list = [] #框中心y坐标
    center_list = [] #框中心坐标
    for i in range(len(recs)):
        rec = recs[i]
        center_x, center_y = _get_rec_center(rec)
        center_x_list.append(center_x)
        center_y_list.append(center_y)
    return center_x_list, center_y_list

# 重排rec顺序
# 原始rec四点顺序都是按逆时针给出，但是起点可能错误
# 每个rec为4点坐标，对应四条边。找出两组对边，比较两组对边中最长边
# 选择最长边所在对边组，长边中心y较小的在上方，以此边为开始的一条边重排rec四点顺序
# 1月11日更新：发现部分图片中，边框的宽度方向的长度可能小于高度方向
# 考虑一种新的重排方法，即参考多个边框的中心得到整体走向，根据这个走向来重排
# 现在可以先做简单的重排，即认为端子标签不是特别倾斜，四边向量与图片宽度方向夹角小的为W，另外的为H
def reorder_rec_my_version(rec):
    # 认定W为长边， H为短边
    W1, W2, H1, H2 = _get_side_vector(rec)
    # 每条边的中心
    center_W1, center_W2, center_H1, center_H2 = ((rec[0] + rec[2]) / 2, (rec[1] + rec[3]) / 2), \
                                                 ((rec[6] + rec[4]) / 2, (rec[7] + rec[5]) / 2), \
                                                 ((rec[2] + rec[4]) / 2, (rec[3] + rec[5]) / 2), \
                                                 ((rec[0] + rec[6]) / 2, (rec[1] + rec[7]) / 2)
    four_sides = [W1, W2, H1, H2]
    angle_sides = []
    # length_sides = []
    for j in range(4):
        angle_sides.append(atan(abs(four_sides[j][1]) / (abs(four_sides[j][0]) + epsilon)))
        # length_sides.append(np.linalg.norm(four_sides[j]))
    # 说明W1，W2为最长边所在的一组对边
    # if max(length_sides[0], length_sides[1]) >= max(length_sides[2], length_sides[3]): 
    if (angle_sides[0] + angle_sides[1]) / 2 <= (angle_sides[2] + angle_sides[3]) / 2: 
        # 选择中心y在上方的作为起始边重排顺序
        if center_W1[1] < center_W2[1]:
            reordered_rec = [rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7]]    
            return reordered_rec, '0,1,2,3'
        else:
            reordered_rec = [rec[4], rec[5], rec[6], rec[7], rec[0], rec[1], rec[2], rec[3]]
            return reordered_rec, '2,3,0,1'
    else:
        if center_H1[1] < center_H2[1]:
            reordered_rec = [rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[0], rec[1]]
            return reordered_rec,'1,2,3,0'
        else:
            reordered_rec = [rec[6], rec[7], rec[0], rec[1], rec[2], rec[3], rec[4], rec[5]]
            return reordered_rec,'3,0,1,2'

# 参照作者代码
def reorder_rec(rec):
    rec = np.array(rec).reshape((4, 2))
    rec = reorder_vertexes(rec)
    # 调整起点
    tmp_x, tmp_y = rec[3, 0], rec[3, 1]
    for i in range(3, 0, -1):
        rec[i] = rec[i - 1]
    rec[0, 0], rec[0, 1] = tmp_x, tmp_y
    rec = np.reshape(rec, (1, 8))[0]
    return rec, 'test'

#  生成rec信息
# 输入为重排后的rec坐标
# 目前考虑如下信息：中心坐标2维、W长度2维、H长度2维、W与图片宽度方向夹角2维、H与图片宽度方向夹角2维
def generate_rec_data(rec):
    tmp_rec, _ = reorder_rec(rec)
    center_x, center_y = _get_rec_center(tmp_rec)
    W1, W2, H1, H2 = _get_side_vector(tmp_rec)
    four_sides = [W1, W2, H1, H2]   
    length_sides = []
    for j in range(4):
        length_sides.append(np.linalg.norm(four_sides[j]))
    # length_W, length_H = (length_sides[0] + length_sides[1]) / 2, (length_sides[2] + length_sides[3]) / 2 
    # rotate_angle_W = (atan(W1[1] / W1[0]) + atan(W2[1] / W2[0])) / 2
    # rotate_angle_H = (atan(H1[1] / H1[0]) + atan(H2[1] / H2[0])) / 2
    length_W1, length_W2 = length_sides[0], length_sides[1]
    length_H1, length_H2 = length_sides[2], length_sides[3]
    rotate_angle_W1, rotate_angle_W2 = atan(W1[1] / (W1[0] + epsilon)), atan(W2[1] / (W2[0] + epsilon))
    if atan(H1[1] / (H1[0] + epsilon)) > 0:
        rotate_angle_H1 = pi + atan(H1[1] / (H1[0] + epsilon))
    else:
        rotate_angle_H1 = atan(H1[1] / (H1[0] + epsilon))
    if atan(H2[1] / (H2[0] + epsilon)) > 0:
        rotate_angle_H2 = pi + atan(H2[1] / (H2[0] + epsilon))
    else:
        rotate_angle_H2 = atan(H2[1] / (H2[0] + epsilon))
    rec_data = [center_x, center_y,
                length_W1, length_W2,
                length_H1, length_H2,
                rotate_angle_W1, rotate_angle_W2,
                rotate_angle_H1, rotate_angle_H2]
    return rec_data
   
# 得到某一维rec_data的平均值
# 单独计算data > 0 的平均和 data < 0的平均，最终符号以计数较多为准
def get_sign_avg_rec_data(rec_data_list):
    positive = [x for x in rec_data_list if x > 0]
    negative = [x for x in rec_data_list if x < 0]
    if len(positive) > len(negative):
        if len(positive) != 0:
            return np.mean(positive)
    else:
        if len(negative) != 0:
            return np.mean(negative)
# 对坐标求LOF
# 输入：框四点坐标list
# 返回：内围点框四点坐标list
# 首先输入所有recs，重排后生成rec信息
# 以中位数为基准归一化，进行LOF检测
def LOF(recs_all):
    np.set_printoptions(suppress=True)
    cnt_recs = len(recs_all)
    center_x_flag, center_y_flag = False, False
    length_W_flag, length_H_flag = True, True
    rotate_angle_W_flag, rotate_angle_H_flag = True, True
    coefficients = [1, 1, 1.5, 1, 1, 1.5]
    # W长度之差1维，H长度平均值1维，W旋转角之差1维，H旋转角平均值1维
    center_x_list, center_y_list = [], []
    length_W_list, length_H_list = [], []
    rotate_angle_W_list, rotate_angle_H_list = [], []
    rec_data_list = []
    if center_x_flag:
        rec_data_list.append(center_x_list)
    if center_y_flag:
        rec_data_list.append(center_y_list)
    if length_W_flag:
        rec_data_list.append(length_W_list)
    if length_H_flag:
        rec_data_list.append(length_H_list)
    if rotate_angle_W_flag:
        rec_data_list.append(rotate_angle_W_list)
    if rotate_angle_H_flag:
        rec_data_list.append(rotate_angle_H_list)
    cnt_data_categories = len(rec_data_list)   
    
    # 重排并生成rec_data列表
    for i in range(cnt_recs):
        reordered_rec, _ = reorder_rec(recs_all[i])
        rec_data = generate_rec_data(reordered_rec)
          
        if center_x_flag:
            center_x_list.append(coefficients[0] * rec_data[0])
        if center_y_flag:
            center_y_list.append(coefficients[1] * rec_data[1])
        if length_W_flag:
            length_W_list.append(coefficients[2] * (rec_data[2] + rec_data[3]) / 2)
        if length_H_flag:
            length_H_list.append(coefficients[3] * (rec_data[4] + rec_data[5]) / 2)
        if rotate_angle_W_flag:
            rotate_angle_W_list.append(coefficients[4] * (rec_data[6] - rec_data[7]))
        if rotate_angle_H_flag:
            rotate_angle_H_list.append(coefficients[5] * (rec_data[8] + rec_data[9]) / 2)
        
        
    # print(rec_data_list)
    # 使用preprocessing模块中的StandardScaler取代归一化
    # 对rec_data列表归一化
    # for i in range(cnt_data_categories):
    #     # 获取中位数
    #     # 尝试计算平均值与标准差，然后使用筛选后的平均值作为基准进行归一化
    #     # median = np.median(rec_data_list[i])
    #     avg_before_filter = np.mean(rec_data_list[i])
    #     std = np.sqrt(np.mean((rec_data_list[i] - np.mean(rec_data_list[i]))**2))
    #     rec_data_after_filter = [x for x in rec_data_list[i] if abs(x - avg_before_filter) < 3 * std]
    #     avg_after_filter = np.mean(rec_data_after_filter)
    #     # 归一化
    #     # 尝试是否归一化对LOF结果的影响
    #     for j in range(len(rec_data_list[i])):
    #         rec_data_list[i][j] /= avg_after_filter
    #         #rec_data_list[i][j] /= median
    # # 转为np.array，形状为cnt_data_categories行 X cnt_recs列
    # 转置为(n_samples, n_features)
    rec_data_list = np.array(rec_data_list).T
    # rec_data_list = RobustScaler().fit_transform(rec_data_list)
    # rec_data_list = normalize(rec_data_list, norm = 'l2', axis = 0)
    rec_data_list = StandardScaler().fit_transform(rec_data_list)
    # print(rec_data_list)
    clf = LocalOutlierFactor(n_neighbors = max(int(0.3 * cnt_recs), 1))
    LOF_data = clf.fit_predict(rec_data_list)
    # print(clf.negative_outlier_factor_)
    # return LOF_data
    return clf.negative_outlier_factor_, LOF_data, rec_data_list
  
 # PCA

# 10/3完成
# 输入：框四点坐标list
# 返回：分类信息
def PCA_(recs_all, img_width, img_height):
    np.set_printoptions(suppress=True)
    width, height = img_width, img_height
    cnt_recs = len(recs_all)
    center_x_flag, center_y_flag = True, True
    length_W_flag, length_H_flag = False, False
    rotate_angle_W_flag, rotate_angle_H_flag = False, False
    coefficients = [1, 1, 1, 1, 1, 1]
    # 中心坐标2维，W长度之差1维, H长度之差1维
    center_x_list, center_y_list = [], []
    length_W_list, length_H_list = [], []
    rotate_angle_W_list, rotate_angle_H_list = [], []
    rec_data_list = []
    if center_x_flag:
        rec_data_list.append(center_x_list)
    if center_y_flag:
        rec_data_list.append(center_y_list)
    if length_W_flag:
        rec_data_list.append(length_W_list)
    if length_H_flag:
        rec_data_list.append(length_H_list)
    if rotate_angle_W_flag:
        rec_data_list.append(rotate_angle_W_list)
    if rotate_angle_H_flag:
        rec_data_list.append(rotate_angle_H_list)
    cnt_data_categories = len(rec_data_list)   
    
    # 填充rec信息
    for i in range(cnt_recs):
        reordered_rec, _ = reorder_rec(recs_all[i])
        rec_data = generate_rec_data(reordered_rec)
        if center_x_flag:
            center_x_list.append(coefficients[0] * rec_data[0] / width)
        if center_y_flag:
            center_y_list.append(coefficients[1] * rec_data[1] / height)
        # 使用W或H的平均值
        if length_W_flag:
            length_W_list.append(coefficients[2] * (rec_data[2] + rec_data[3]) / width)
        if length_H_flag:
            length_H_list.append(coefficients[3] * (rec_data[4] + rec_data[5]) / height)
        if rotate_angle_W_flag:
            rotate_angle_W_list.append(coefficients[4] * (rec_data[6] - rec_data[7]))
        # 使用H旋转角的平均值   
        if rotate_angle_H_flag:
            rotate_angle_H_list.append(coefficients[5] * (rec_data[8] + rec_data[9]) / 2 / (pi / 2))
    # 转置为(n_samples, n_features)
    rec_data_list = np.array(rec_data_list).T
    # rec_data_list = StandardScaler().fit_transform(rec_data_list)
    # rec_data_list = MinMaxScaler().fit_transform(rec_data_list)
    pca_ = PCA(n_components = 1).fit(rec_data_list)
    new_rec_data_list = pca_.transform(rec_data_list)
    # print(clf.negative_outlier_factor_)
    # return LOF_data
    return pca_, new_rec_data_list

# 由rec列表得到rec之差列表,delta_order为差的阶数
# 例:delta_order = n, delta_rec = rec(i + n) - rec(i)
# 输入为PCA降维后的rec_list
def get_delta_rec_list(delta_order, pca_rec_list):
    cnt_rec = len(pca_rec_list)
    delta_rec_list = []
    if delta_order >= cnt_rec:
        return None
    else:
        for i in range(delta_order, cnt_rec):
            delta_rec = pca_rec_list[i] - pca_rec_list[i - delta_order]
            delta_rec_list.append(delta_rec)
        return delta_rec_list

# 对rec列表分组
# 输入为1阶delta_rec，PCA降维后rec_list
# 输出为各组rec索引
def _divide_rec_list(avg_delta_one_rec, avg_delta_two_rec, pca_rec_list):
    terminal_distribution_types = ('2cols', '1col_1class', '1col_nclasses')
    cnt_delta_one_rec_mutation = 0
    rec_group_list = []
    # 分析terminal_distribution_type = tdt
    # 对于单列端子,avg_delta_two_rec大致等于2 * avg_delta_one_rec
    # 所以不符合这种情况的就是两列端子
    if abs(avg_delta_two_rec) > 2.5 * abs(avg_delta_one_rec) or abs(avg_delta_two_rec) < abs(avg_delta_one_rec):
        # tdt = terminal_distribution_type
        tdt = terminal_distribution_types[0]
        tmp_rec_list1, tmp_rec_list2 = [], []
        for i in range(len(pca_rec_list)):
            # 对于两列端子，PCA后分别是负号和正号
            if pca_rec_list[i] > 0:
                tmp_rec_list1.append(i)
            else:
                tmp_rec_list2.append(i)
        rec_group_list.append(tmp_rec_list1)
        rec_group_list.append(tmp_rec_list2)
        return tdt, rec_group_list, avg_delta_one_rec, avg_delta_one_rec
    else:
        # rec数过少时可能会被大的delta_one_rec提高平均值
        # 因此过滤后计算
        delta_one_rec_data_list = np.array(get_delta_rec_list(1, pca_rec_list))
        std = np.sqrt(np.mean((delta_one_rec_data_list - avg_delta_one_rec) ** 2))
        avg_after_filter = np.mean([x for x in delta_one_rec_data_list if abs(x - avg_delta_one_rec) < 2.5 * std])
        # 计算1阶delta_rec突变数目
        tmp_rec_list = []
        tmp_rec_list.append(0)
        for i in range(1, len(pca_rec_list)):
            # 如果大于1.5倍的avg_delta_one_rec， 就认为是突变
            if abs(pca_rec_list[i] - pca_rec_list[i - 1]) > 1.6 * abs(avg_after_filter):
                cnt_delta_one_rec_mutation += 1
                # 将之前的储存到返回列表中，清零当前rec列表为下一类做准备
                _ = tmp_rec_list.copy()
                rec_group_list.append(_)
                tmp_rec_list.clear()
            tmp_rec_list.append(i)
        rec_group_list.append(tmp_rec_list)
        if cnt_delta_one_rec_mutation == 0:
            tdt = terminal_distribution_types[1]
        else:
            tdt = terminal_distribution_types[2]
        return tdt, rec_group_list, avg_after_filter, std

#  对rec列表分组，集合上述功能
def divide_rec_list(pca_rec_list):
    # rec_i - rec_i-1， cnt - 1个数据
    delta_one_rec_data_list = get_delta_rec_list(1, pca_rec_list)
    # rec_i - rec_i-2， cnt - 2个数据
    delta_two_rec_data_list = get_delta_rec_list(2, pca_rec_list)
    avg_delta_one_rec = sum(delta_one_rec_data_list) / len(delta_one_rec_data_list)
    avg_delta_two_rec = sum(delta_two_rec_data_list) / len(delta_two_rec_data_list)
    # 对rec分组，返回每组rec索引
    tdt, rec_group_list, avg_after_filter, std = _divide_rec_list(avg_delta_one_rec, avg_delta_two_rec, pca_rec_list) 
    return tdt, rec_group_list, avg_after_filter, std
# 聚类
# 输入：框四点坐标list
# 返回：分类信息
def DBSCAN_(recs_all):
    np.set_printoptions(suppress=True)
    cnt_recs = len(recs_all)
    center_x_flag, center_y_flag = True, False
    length_W_flag, length_H_flag = False, False
    rotate_angle_W_flag, rotate_angle_H_flag = False, False
    coefficients = [1, 1, 1, 1, 1, 1]
    # 中心坐标2维，W长度之差1维, H长度之差1维
    center_x_list, center_y_list = [], []
    length_W_list, length_H_list = [], []
    rotate_angle_W_list, rotate_angle_H_list = [], []
    rec_data_list = []
    if center_x_flag:
        rec_data_list.append(center_x_list)
    if center_y_flag:
        rec_data_list.append(center_y_list)
    if length_W_flag:
        rec_data_list.append(length_W_list)
    if length_H_flag:
        rec_data_list.append(length_H_list)
    if rotate_angle_W_flag:
        rec_data_list.append(rotate_angle_W_list)
    if rotate_angle_H_flag:
        rec_data_list.append(rotate_angle_H_list)
    cnt_data_categories = len(rec_data_list)   
    
    # 填充rec信息
    for i in range(cnt_recs):
        reordered_rec, _ = reorder_rec(recs_all[i])
        rec_data = generate_rec_data(reordered_rec)
        center_x_list.append(coefficients[0] * rec_data[0])
        center_y_list.append(coefficients[1] * rec_data[1])
        # W差值与H平均值
        length_W_list.append(coefficients[2] * (rec_data[2] - rec_data[3]))
        length_H_list.append(coefficients[3] * (rec_data[4] + rec_data[5]))
        rotate_angle_W_list.append(coefficients[4] * (rec_data[6] - rec_data[7]))
        rotate_angle_H_list.append(coefficients[5] * (rec_data[8] + rec_data[9]) / 2)
    # 转置为(n_samples, n_features)
    rec_data_list = np.array(rec_data_list).T
    # rec_data_list = StandardScaler().fit_transform(rec_data_list)
    # rec_data_list = MinMaxScaler().fit_transform(rec_data_list)
    cluster = DBSCAN(eps = 0.15, min_samples = max(1, int(0.05 * cnt_recs))).fit(rec_data_list)
    DBSCAN_data = cluster.labels_
    # print(clf.negative_outlier_factor_)
    # return LOF_data
    return DBSCAN_data, rec_data_list

# x坐标和y坐标的预回归
def _pre_regression_on_center_x_y(center_x_list, center_y_list):
    delta_x = (max(center_x_list) - min(center_x_list))
    delta_y = (max(center_y_list) - min(center_y_list))
    ratio = [(tmp_y - center_y_list[0]) / delta_y for tmp_y in center_y_list]
    var_x = np.array(center_x_list).T.reshape(-1, 1)
    var_y = np.array(center_y_list)
    reg = LinearRegression().fit(var_x, var_y)
    k = reg.coef_[0]
    if k > 0:
        new_center_x_list = [min(center_x_list) + tmp_ratio * delta_x for tmp_ratio in ratio]
    else:
        new_center_x_list = [max(center_x_list) - tmp_ratio * delta_x for tmp_ratio in ratio]
    return new_center_x_list
# 回归
def regression_(recs_all):
    np.set_printoptions(suppress=True)
    cnt_recs = len(recs_all)
    center_x_flag, center_y_flag = True, True
    length_W_flag, length_H_flag = True, True
    rotate_angle_W_flag, rotate_angle_H_flag = True, False
    coefficients = [1, 1, 1, 1, 1, 1]
    # 中心坐标2维，W长度之差1维, H长度之差1维
    center_x_list, center_y_list = [], []
    length_W_list, length_H_list = [], []
    rotate_angle_W_list, rotate_angle_H_list = [], []
    for i in range(cnt_recs):
        reordered_rec, _ = reorder_rec(recs_all[i])
        rec_data = generate_rec_data(reordered_rec)
        if center_x_flag:
            center_x_list.append(coefficients[0] * rec_data[0])
        if center_y_flag:
            center_y_list.append(coefficients[1] * rec_data[1])
        # 使用W或H的平均值
        if length_W_flag:
            length_W_list.append(coefficients[2] * (rec_data[2] + rec_data[3]) / 2)
        if length_H_flag:
            length_H_list.append(coefficients[3] * (rec_data[4] + rec_data[5]) / 2)
        if rotate_angle_W_flag:
            rotate_angle_W_list.append(coefficients[4] * (rec_data[6] + rec_data[7]) / 2)
        # 使用H旋转角的平均值   
        if rotate_angle_H_flag:
            rotate_angle_H_list.append(coefficients[5] * (rec_data[8] + rec_data[9]) / 2)
    # 转置为(n_samples, n_features)
    # rec_data_list = np.array(rec_data_list).T
    # TODO:当接近垂直排布时，x的坐标非常接近，回归方程的截距很大，容易出现误差
    # 可能不是均匀分布的，根据y的分布来重新计算x的值
    # 先进行一次关于y和x的预回归
    new_center_x_list = _pre_regression_on_center_x_y(center_x_list, center_y_list)
    center_list = [new_center_x_list, center_y_list]
    center_x_var_list = np.array(new_center_x_list).T.reshape(-1, 1)
    center_y_var_list = np.array(center_y_list).T
    X_var_list = np.array([new_center_x_list, center_y_list]).T
    y_var1_list = np.array(length_W_list).T
    y_var2_list = np.array(length_H_list).T
    y_var3_list = np.array(rotate_angle_W_list).T

    # reg_center = LinearRegression().fit(center_x_var_list, center_y_var_list)
    reg_length_W = LinearRegression().fit(X_var_list, y_var1_list)
    reg_length_H = LinearRegression().fit(X_var_list, y_var2_list)
    reg_rotate_angle_W = LinearRegression().fit(X_var_list, y_var3_list)
    # print(clf.negative_outlier_factor_)
    # return LOF_data
    # regression_line_center = (reg_center.coef_, reg_center.intercept_)
    regression_line_length_W = (reg_length_W.coef_, reg_length_W.intercept_)
    regression_line_length_H = (reg_length_H.coef_, reg_length_H.intercept_)
    regression_line_rotate_angle_W = (reg_rotate_angle_W.coef_, reg_rotate_angle_W.intercept_)
    # return regression_line_center, regression_line_length_W, regression_line_length_H, regression_line_rotate_angle_W
    return center_list, regression_line_length_W, regression_line_length_H, regression_line_rotate_angle_W


# 根据回归中心坐标直线得到回归直线上离原始坐标最近的点
# rec_point为(中心x坐标，中心y坐标)
def get_closest_point(regression_line, rec_point):
    # y = coef_ * x + intercept_ = kx + b
    # 法向量为(coef_, -1)
    # 设最近点为(x + coef_ * t, y - t)， 代入回归平面方程
    k, b = regression_line[0][0], regression_line[1]
    x, y = rec_point[0], rec_point[1]
    t = (-k * x + y - b) / (k ** 2 + 1)
    closest_rec_point = (x + k * t, y - t)
    return closest_rec_point

# 根据重新分布后的center_list得到center
def get_center_x_by_list(rec_center, center_list):
    center_x_list, center_y_list = center_list[0], center_list[1]
    x, y = rec_center[0], rec_center[1]
    n = len(center_x_list) - 1
    xn, x0 = center_x_list[n], center_x_list[0]
    yn, y0 = center_y_list[n], center_y_list[0]
    new_x = (y - y0) * (xn - x0) / (yn - y0) + x0
    return new_x
# TODO:这个函数有必要拆成更基础的函数
# 根据回归得到新的rec_data
# rec_data = [中心x，中心y，length_W，length_H, angle_W]
def get_new_rec_data_by_regression(rec, center_list, regression_length_W, regression_length_H, regression_rotate_angle_W):
    rec_data = generate_rec_data(rec)
    rec_center = _get_rec_center(rec)
    # new_rec_center = get_closest_point(regression_center, rec_center)
    # x, y = new_rec_center[0], new_rec_center[1]
    x = get_center_x_by_list(rec_center, center_list)
    y = rec_center[1]
    # old_x, old_y = rec_center[0], rec_center[1]
    coef_length_W, intercept_length_W = regression_length_W[0], regression_length_W[1]
    coef_length_H, intercept_length_H = regression_length_H[0], regression_length_H[1]
    coef_angle,  intercept_angle  = regression_rotate_angle_W[0], regression_rotate_angle_W[1]
    new_length_W = coef_length_W[0] * x + coef_length_W[1] * y + intercept_length_W
    new_length_H = coef_length_H[0] * x + coef_length_H[1] * y + intercept_length_H
    new_rotate_angle_W = coef_angle[0] * x + coef_angle[1] * y + intercept_angle
    # 与上文rec_data格式不一样， 其中的长度和角度均取平均值
    new_rec_data = [x, y, new_length_W, new_length_H, new_rotate_angle_W]
    return new_rec_data

# 从rec_data得到四点坐标
# rec_data = [中心x，中心y，length_W，length_H, angle_W, angle_H1, angle_H2]
# 根据四边向量和中心得到四点坐标可以写成函数
def get_four_point_by_rec_data(rec_data):
    rec_center = rec_data[0], rec_data[1]
    length_W, length_H = rec_data[2], rec_data[3]
    angle_W = rec_data[4]
    angle_H1, angle_H2 = rec_data[5], rec_data[6]
    vector_W  = (length_W * cos(angle_W),  length_W * sin(angle_W))
    vector_H1 = (length_H * cos(angle_H1), length_H * sin(angle_H1))
    vector_H2 = (length_H * cos(angle_H2), length_H * sin(angle_H2))
    vector_H1, vector_H2 = _unify_length(vector_H1, vector_H2)
    rec_center, vector_W, vector_H1, vector_H2 = np.array(rec_center), np.array(vector_W), np.array(vector_H1), np.array(vector_H2)
    # l/r/t/b = left/right/top/bottom
    rt = rec_center + 0.5 * vector_W + 0.5 * vector_H2
    lt = rec_center - 0.5 * vector_W + 0.5 * vector_H1
    rb = rec_center + 0.5 * vector_W - 0.5 * vector_H2
    lb = rec_center - 0.5 * vector_W - 0.5 * vector_H1
    rec = (rt[0], rt[1], lt[0], lt[1], lb[0], lb[1], rb[0], rb[1])
    return rec

#  校正框
# 输入:筛选掉异常点的框四点坐标list
# 返回：根据框中心坐标生成的框四点坐标list
def recs_correction(recs_all, img_width, img_height):
    width, height = img_width, img_height
    pca_, rec_pca_list = PCA_(recs_all, width, height)
    if len(rec_pca_list) < 3:
        return recs_all
    tdt, rec_index_group_list, avg_after_filter, std = divide_rec_list(rec_pca_list)
    new_recs_all = recs_all.copy()
    for i in range(len(rec_index_group_list)):
        rec_index_group = rec_index_group_list[i]
        if len(rec_index_group) < 3:
            # 个数太少，不进行数据分析
            # TODO:加入根据回归直线合并操作
            continue
        else:
            # 生成当前group中所有rec的四点坐标
            rec_group = [recs_all[x] for x in rec_index_group]
            # 筛选异常点
            factors, _, _ = LOF(rec_group)
            j = 0
            recs_after_LOF = []
            rotate_angle_H1_list, rotate_angle_H2_list = [], []
            for rec in rec_group:
                if abs(factors[j]) < 1.6:
                    recs_after_LOF.append(rec)
                    rec_data = generate_rec_data(rec)
                    rotate_angle_H1_list.append(rec_data[8])
                    rotate_angle_H2_list.append(rec_data[9])
                j += 1                  
            # 对剩下的点最小二乘回归，计算W长度与倾斜角W随x，y的关系
            rotate_angle_H1, rotate_angle_H2 = get_sign_avg_rec_data(rotate_angle_H1_list), get_sign_avg_rec_data(rotate_angle_H2_list)                
            
            # center_list, regression_length_W, regression_length_H, regression_rotate_angle_W = regression_(recs_after_LOF)
            center_list, regression_length_W, regression_length_H, regression_rotate_angle_W = regression_(recs_after_LOF)            
            j = 0
            for rec in rec_group:
                new_rec_data = get_new_rec_data_by_regression(rec, center_list, regression_length_W, regression_length_H, regression_rotate_angle_W)
                new_rec_data.append(rotate_angle_H1)
                new_rec_data.append(rotate_angle_H2)
                new_rec = get_four_point_by_rec_data(new_rec_data)
                new_recs_all[rec_index_group[j]] = new_rec
                j += 1
    return new_recs_all, rec_index_group_list

# 从框四点坐标得到数字区域坐标
# 输入：一张图片所有recs
# 返回：每个rec的数字部分坐标
# 主要是通过取中心
# 
def get_digit_area(rec):
    # 通过向量加减得到四点
    # 四点比例系数 rt/lt/rb/lb
    rec_center = _get_rec_center(rec)
    vector_W1, vector_W2, vector_H1, vector_H2 = _get_side_vector(rec)
    angle_W1, angle_W2 = atan(vector_W1[1] / (vector_W1[0] + epsilon)), atan(vector_W2[1] / (vector_W2[0] + epsilon))
    ratio = 0.1 * sin(angle_W1)
    coefs = 0.5 * np.array([[0.45 + ratio, 0.60 + ratio], [0.50 + 2 * ratio, 0.60 + ratio], [0.50 + ratio, 0.50 + ratio], [0.50 + 2 * ratio, 0.50 + ratio]])
    vector_W1, vector_W2 = _unify_length(vector_W1, vector_W2)
    vector_H1, vector_H2 = _unify_length(vector_H1, vector_H2)
    rec_center, vector_W1, vector_W2, vector_H1, vector_H2 = np.array(rec_center), np.array(vector_W1), np.array(vector_W2), np.array(vector_H1), np.array(vector_H2)
    vector_W = (vector_W1 + vector_W2) / 2
    # 顺应方向选择数字区域为平行四边形
    if angle_W1 > 0:
        vector_H = vector_H2
    else:
        vector_H = vector_H1
    rt = rec_center + coefs[0][0] * vector_W + coefs[0][1] * vector_H
    lt = rec_center - coefs[1][0] * vector_W + coefs[1][1] * vector_H
    rb = rec_center + coefs[2][0] * vector_W - coefs[2][1] * vector_H
    lb = rec_center - coefs[3][0] * vector_W - coefs[3][1] * vector_H
    digit_area = (rt[0], rt[1], lt[0], lt[1], lb[0], lb[1], rb[0], rb[1])
    return digit_area

def get_all_digit_areas(recs_all):
    digit_areas = []
    for rec in recs_all:
        digit_area = get_digit_area(rec)
        digit_areas.append(digit_area)
    return digit_areas

# 返回position center位于哪一段区间
def find_center_corresponding_position(position_center, sequential_position):
    for i in range(len(sequential_position)):
        position = sequential_position[i]
        if position[0] < position_center < position[1]:
            return i
    return 999

# 每个group中的所有编号是顺序的，所以使用编号识别结果减去索引应该是一致的
# 依此进行矫正
def digit_results_correction(sequential_digit_position, recognize_digit_and_position, previous_digit_cnt):
     
    start_numbers_all = {}
    # 检测到有digit的区域个数，但并不是每个digit都能够识别
    cnt_digit = len(sequential_digit_position)
    for key in recognize_digit_and_position.keys():
        digit = int(key[:-1])
        position = recognize_digit_and_position[key]
        center = (position[0] + position[1]) / 2
        index = find_center_corresponding_position(center, sequential_digit_position)
        start_number = str(digit - index - previous_digit_cnt)
        if start_number not in start_numbers_all.keys():
            start_numbers_all[start_number] = 1
        else:
            start_numbers_all[start_number] += 1
    # 出现次数最多的作为最终的start_number
    final_start_number = int(max(zip(start_numbers_all.values(), start_numbers_all.keys()))[1])

    digit_results_corrected = []
    for i in range(cnt_digit):
        digit_results_corrected.append(i + final_start_number + previous_digit_cnt)
    return digit_results_corrected