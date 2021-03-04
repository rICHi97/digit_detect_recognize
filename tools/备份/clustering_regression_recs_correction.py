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
def reorder_vertexes(xy_list):
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

# from copy import deepcopy
# 得到各边向量
# 返回顺序为W1，W2， H1，H2
# 从起始边逆时针顺序
def _get_side_vector(rec):
    return ((rec[0] - rec[2], rec[1] - rec[3]), 
            (rec[6] - rec[4], rec[7] - rec[5]), 
            (rec[2] - rec[4], rec[3] - rec[5]), 
            (rec[0] - rec[6], rec[1] - rec[7]))

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


# 生成rec信息
# 输入为重排后的rec坐标
# 目前考虑如下信息：中心坐标2维、W长度2维、H长度2维、W与图片宽度方向夹角2维、H与图片宽度方向夹角2维
def generate_rec_data(rec):
    center_x, center_y = _get_rec_center(rec)
    W1, W2, H1, H2 = _get_side_vector(rec)
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
    rotate_angle_H1, rotate_angle_H2 = atan(H1[1] / (H1[0] + epsilon)), atan(H2[1] / (H2[0] + epsilon))
    rec_data = [center_x, center_y,
                length_W1, length_W2,
                length_H1, length_H2,
                rotate_angle_W1, rotate_angle_W2,
                rotate_angle_H1, rotate_angle_H2]
    return rec_data
    


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
# 聚类
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
def divide_rec_list(avg_delta_one_rec, avg_delta_two_rec, pca_rec_list):
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

# 回归
def regression_(recs_all):
    np.set_printoptions(suppress=True)
    cnt_recs = len(recs_all)
    center_x_flag, center_y_flag = True, True
    length_W_flag, length_H_flag = True, False
    rotate_angle_W_flag, rotate_angle_H_flag = True, False
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
    X_var_list = [center_x_list, center_y_list]
    y_var1_list = [length_W_list]
    y_var2_list = [rotate_angle_W_list]
    # 填充rec信息
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
    rec_data_list = np.array(rec_data_list).T
    X_var_list = np.array(X_var_list).T
    y_var1_list = np.array(y_var1_list).T
    y_var2_list = np.array(y_var2_list).T
    # rec_data_list = StandardScaler().fit_transform(rec_data_list)
    # rec_data_list = MinMaxScaler().fit_transform(rec_data_list)
    reg1 = LinearRegression().fit(X_var_list, y_var1_list)
    reg2 = LinearRegression().fit(X_var_list, y_var2_list)
    # print(clf.negative_outlier_factor_)
    # return LOF_data
    regression_line1 = (reg1.coef_, reg1.intercept_)
    regression_line2 = (reg2.coef_, reg2.intercept_)
    return regression_line1, regression_line2

# 得到回归rec_data信息
# 从中心点计算得到回归平面上最近的点
# rec_point为(中心x坐标，中心y坐标，待)
def get_closest_point(regression_line, rec_point):
    # y = coef_[0] * x0 + coef_[1] * x1 + intercept_
    # 法向量为(coef_[0], coef_[1], -1)
    # 设最近点为(x0 + coef_[0] * t, x1 + coef_[1] * t, y - t)， 代入回归平面方程
    coef, intercept = regression_line[0], regression_line[1]
    beta0, beta1 = coef[0], coef[1]
    x0, x1, y = rec_point[0], rec_point[1], rec_point[2]
    t = (-beta0 * x0 + -beta1 * x1 + y - intercept) / (beta0 ** 2 + beta1 ** 2 + 1)
    closest_rec_point = (x0 + beta0 * t, x1 + beta1 * t, y - t)
    return closest_rec_point

# 校正框
# 输入:筛选掉异常点的框四点坐标list
# 返回：根据框中心坐标生成的框四点坐标list
def recs_correction(recs_all):
    cnt_recs = len(recs_all)
    recs_reordered_all = []
    recs_corrected_all = []
    coefficient_W, coefficient_H = 0.2, 0.3
    # 重排点顺序
    # 后序可能需要调整，确保直接输入已重排好的rec
    for i in range(cnt_recs):
        reordered_rec, order_str = reorder_rec(recs_all[i])
        recs_reordered_all.append(reordered_rec)
    # 矫正框为平行四边形  
    for i in range(cnt_recs):
        rec = recs_reordered_all[i]
        center_x, center_y = _get_rec_center(rec)
        W1, W2, H1, H2 = _get_side_vector(rec)
        avg_W_vector, avg_H_vector = np.array([(W1[0] + W2[0]) / 2, (W1[1] + W2[1]) / 2]), np.array([(H1[0] + H2[0]) / 2, (H1[1] + H2[1]) / 2])
        center_vector = np.array([center_x, center_y])
        # half_W_vector, half_H_vector = [avg_W[0] / 2, avg_W[1] / 2], [avg_H[0] / 2, avg_H[1] / 2]
        # udlr = up down left right
        ur_vector = center_vector + 0.95 * coefficient_W * avg_W_vector + coefficient_H * avg_H_vector
        ul_vector = center_vector - 1.05 * coefficient_W * avg_W_vector + coefficient_H * avg_H_vector
        dl_vector = center_vector - 1.05 * coefficient_W * avg_W_vector - 0.9 * coefficient_H * avg_H_vector
        dr_vecror = center_vector + 0.95 * coefficient_W * avg_W_vector - 0.9 * coefficient_H * avg_H_vector
        number_region = [ur_vector[0], ur_vector[1], ul_vector[0], ul_vector[1], dl_vector[0], dl_vector[1], dr_vecror[0], dr_vecror[1]]
        recs_corrected_all.append(number_region)
    # return recs_reordered_all
    return recs_corrected_all
    
def number_results_correction(numbers):
     # print(numbers)
     NUMBER_THRESHOLD = 50
     numbers_indices_list = []
     numbers_indices_valid_list = []
     avg_number = 0
     cnt_invalid_number = 0 
     # avg_start_number = 0
     # 计算平均值，建立预测数字与序号列表
     
     for i in range(len(numbers)):
        if int(numbers[i]) > NUMBER_THRESHOLD:
            cnt_invalid_number += 1
            continue
        avg_number += int(numbers[i]) / (len(numbers) - cnt_invalid_number)
        # avg_start_number += (numbers[i] - i + 1) / len(numbers)
        numbers_indices_list.append([int(numbers[i]), i])
     # print(numbers_indices_list)
        # print(numbers[i], i)
        
     # 筛选异常值
     # 大于阈值
     for i in range(len(numbers_indices_list)):
         tmp_number = numbers_indices_list[i][0]
         tmp_index = numbers_indices_list[i][1]
         # if tmp_number > NUMBER_THRESHOLD or (tmp_number -  tmp_index) < 0 or tmp_number > (avg_number + int(1.2 * 0.5 * len(numbers))):
         if tmp_number > NUMBER_THRESHOLD or (tmp_number -  tmp_index) < 0:
             continue
         numbers_indices_valid_list.append(numbers_indices_list[i])
     # print(numbers_indices_valid_list)
     
    # 计算起点
     numbers_indices_dict = {}
     for i in range(len(numbers_indices_valid_list)):
         tmp_start_number = numbers_indices_valid_list[i][0] - numbers_indices_valid_list[i][1]
         if str(tmp_start_number) not in numbers_indices_dict:
             numbers_indices_dict[str(tmp_start_number)] = 1
         else:
             numbers_indices_dict[str(tmp_start_number)] += 1
            
     start_number = int(max(zip(numbers_indices_dict.values(), numbers_indices_dict.keys()))[1])
     number_corrected_results = []
     for i in range(len(numbers)):
        number_corrected_results.append(str(start_number + i))
        
     return number_corrected_results
                       
    # avg_lof_score = 0 #平均lof得分
    # inlier_index = [] #内围点序号
    # x_list_after_LOF = [] #LOF之后框中心x坐标
    # recs_after_LOF = [] #LOF之后框四点坐标
    # cnt_recs = len(text_recs_all)
    # # 获取中心点坐标
    # center_x_list, center_y_list = _from_recs_to_centers(text_recs_all)
    # # plt.scatter(x = center_x_list,y = center_y_list)
    
    # # LOF
    # # 只对x坐标计算LOF，注意参数选择
    # # 0.3倍可能过于宽松
    # clf = LocalOutlierFactor(n_neighbors=int(0.4 * cnt_recs))
    # clf.fit(np.array(center_x_list).reshape(-1, 1))
    # lof_scores = clf.negative_outlier_factor_
    # print(lof_scores)
    # for i in range(len(lof_scores)):
    #     avg_lof_score += lof_scores[i] / len(lof_scores)
    # # 绝对值大于均值lof_scores的视为离群点
    # for i in range(len(lof_scores)):
    #     # LOF得分可调，1.6左右
    #     if ( abs(lof_scores[i]) < abs(avg_lof_score) ) or (abs(lof_scores[i]) < 1.6):
    #         inlier_index.append(i)
    #         x_list_after_LOF.append(center_x_list[i])
    #         recs_after_LOF.append(text_recs_all[i])
            
    # return recs_after_LOF
    # return inlier_index

      
# 聚类
# 输入：筛除掉异常点后的框四点坐标list
# 返回：各类的斜率平均值
# 可能会聚类到仅有一个元素的簇，然而在LOF检测后每簇应该有多个元素。
# 对于这种一个元素的簇，可能是原始框坐标有略微的偏差导致其被聚类为单独的一类，这种情况下不将其纳入到求取斜率过程中
# def DBSCAN_Clustering(recs_after_LOF):
#     slope = 0
#     # 获取框中心坐标
#     center_list, center_x_list, center_y_list = _from_recs_to_centers(recs_after_LOF)
#     # x坐标的跨度，点数
#     x_diff, cnt_points = max(center_x_list) - min(center_x_list), len(center_list)
#     # DBSCAN 可以考虑将eps设置为0.4倍宽度 min_samples设置为百分之30的样本数
#     clustering = DBSCAN( eps = 0.4 * x_diff, min_samples = int(0.3 * cnt_points), n_jobs = -1).fit(np.array(center_x_list).reshape(-1, 1))
#     clustering_labels = clustering.labels_
#     print(clustering_labels)
#     unique_labels = np.unique(clustering_labels)
#     # 指的是总共有几类
#     cnt_labels = len(unique_labels)
#     valid_cnt_labels = cnt_labels
#     # 对于每类内的点进行线性回归
#     # cnt_labels为类总数，第i类的label即为i（从0开始)  bug对于异常值-1找不到那一类 这个教训要记住 不能随便以为搞清楚了所有情况
#     # all_clustering_center_list的每个元素即为每一类对应的所有框
#     # clustering_labels[j]即为第j个元素的label
#     all_clustering_center_list = [] 
#     for i in range(cnt_labels):
#         tmp_label = unique_labels[i]
#         # 如果当前label为-1或者只有小于2元素，就去掉这个label
#         if (tmp_label == -1) or (clustering_labels.tolist().count(tmp_label) < 2):
#             valid_cnt_labels -= 1
#             continue
#         clustering_center_list = []
#         for j in range(len(clustering_labels)):
#             # 第j个点类别为tmp_label时，将该点对应的框坐标添加到该类列表中
#             if clustering_labels[j] == tmp_label:
#                 # 之前好像写错了，之前append的是八个点坐标，然后下方只取了左上点回归
#                 clustering_center_list.append(center_list[j])
#                 #clustering_center_list.append(recs_after_LOF[j])
#         all_clustering_center_list.append(clustering_center_list)
#     # print(all_clustering_center_list)    
#     for i in range(valid_cnt_labels):
#         clustering_center_list = all_clustering_center_list[i]
#         clustering_x_list = []
#         clustering_y_list = []
        
#         for j in range(len(clustering_center_list)):
#             clustering_x_list.append(clustering_center_list[j][0])
#             clustering_y_list.append(clustering_center_list[j][1])
#         # 某一类的x坐标相同，为竖线，感觉不太可能
#         if len(np.unique(clustering_x_list)) == 1:
#             print('请略微变换拍摄角度')
#             return
#         # 貌似不能对竖直线回归
#         # 回归后只关注斜率 不关注截距
#         reg = LinearRegression().fit(np.array(clustering_x_list).reshape(-1, 1),
#                                      np.array(clustering_y_list))
#         # print(reg.coef_)
#         # 注意，之前这里又写错了呜呜，应该是除以valid_cnt_labels，即只用有效的聚类数来计算平均斜率
#         slope += reg.coef_ / valid_cnt_labels   
#     print(slope)
#     return slope
        # print(reg.coef_, reg.intercept_)
        # t = np.arange(0, 800, 0.5)
        # x1 = t
        # y1 =r reg.coef_ * x1 + reg.intercept_
        #plt.plot(x1, y1)  
        # plt.ylim(-0.5, 800)
        

    
 

