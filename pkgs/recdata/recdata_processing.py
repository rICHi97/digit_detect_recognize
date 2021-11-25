# -*- coding: utf-8 -*-
"""
Created on 2021-09-06 18:41:29

@author: Li Zhi
"""

"""
本模块用以处理每个端子的预测结果，预测结果表现为一个四边形
TODO：rec(rectangle)应为quad(quadrangel)
xy_list为rec的四顶点坐标list，
vector为rec的四边向量，向量指向为从左至右，从下至上
    -:  ---->
    -:  ^   ^
    -:  ¦   ¦
    -:  ¦   ¦
    -:  ---->
"""
import math

import numpy as np

EPSILON = 1e-4

def _get_vector(length, rotate_angle):

    vector = [length * math.cos(rotate_angle), length * math.sin(rotate_angle)]

    return vector

def _get_vector_length(vector):

    length = np.linalg.norm(vector)

    return length

def _get_vector_rotate_angle(vector):

    rotate_angle = math.atan(vector[1] / vector[0] + EPSILON)

    return rotate_angle


class Recdata(object):
    """
    从rec四点坐标得到rec data，如边长度、中心坐标
    """

    # TODO：好像没法用，对每个rec求data时，data_func的参数无法确定
    # @staticmethod
    # def get_multi_recs_data(recs_xy_list, data_func, **func_args):
    #     """
    #     得到多个rec的data
    #     Parameters
    #     ----------
    #     recs_xy_list：多个rec的四点坐标
    #     data_func：得到每个rec的data的函数

    #     Returns
    #     ----------
    #     recs_data_list: 多个rec的data的list
    #     """    
    @staticmethod
    def get_avg_rec_data(recs_data_list, is_rotate_angle=False):
        """
        对多个端子的rec_data求平均
        单独计算data > 0的平均和data < 0的平均，最终符号以计数较多的为准
        上述操作是由于倾斜角度很小时，很容易在0附近振荡
        Parameters
        ----------
        recs_data_list：多个rec的data列表
        is_rotate_angle：是否为旋转角

        Returns
        ----------
        avg_rec_data：多个rec的data平均
        """
        # TODO：data list长度不能为0
        if is_rotate_angle:
            positive_data = [data for data in recs_data_list if data > 0]   
            negative_data = [data for data in recs_data_list if data < 0]
            if len(positive_data) > len(negative_data):
                avg_rec_data = np.mean(positive_data)
            else:

                avg_rec_data = np.mean(negative_data)
        else:
            avg_rec_data = np.mean(recs_data_list)

        return avg_rec_data

    @staticmethod
    def get_four_edge_vectors(xy_list):
        """
        得到四边向量元组
        返回顺序为：W1，W2，H1，H2
        Parameters
        ----------
        xy_list：rec的4端点坐标

        Returns
        ----------
        edge_vectors：四边向量，按W1 W2 H1 H2顺序
        """
        four_vertexes = RecdataProcess.from_18_to_42(xy_list)
        W1 = [four_vertexes[0][0] - four_vertexes[1][0], four_vertexes[0][1] - four_vertexes[1][1]]
        W2 = [four_vertexes[3][0] - four_vertexes[2][0], four_vertexes[3][1] - four_vertexes[2][1]]
        H1 = [four_vertexes[1][0] - four_vertexes[2][0], four_vertexes[1][1] - four_vertexes[2][1]]
        H2 = [four_vertexes[0][0] - four_vertexes[3][0], four_vertexes[0][1] - four_vertexes[3][1]]
        edge_vectors = [W1, W2, H1, H2]

        return edge_vectors

    @staticmethod
    def get_edge_vector(length, rotate_angle, is_edge_H):
        """
        根据长度和倾斜角获取边向量，倾斜角为atan计算
        需要结合vector形式调整倾斜角
        高度向量的delta_y一定是负数，宽度向量的delta_x一定是正数
        Parameters
        ----------
        length：边向量长度
        rotate_angle：边向量倾斜角
        is_edge_H：是否为高度边

        Returns
        edge_vector：边向量
        ----------
        """
        if is_edge_H:
            if rotate_angle > 0:
                rotate_angle += math.pi
        edge_vector = _get_vector(length, rotate_angle)

        return edge_vector

    @staticmethod
    def get_center(xy_list):
        """
        得到rec中心坐标
        Parameters
        ----------
        xy_list：rec的4端点坐标

        Returns
        ----------
        center_x, center_y：rec中心的xy坐标
        """
        center_x, center_y = 0, 0
        four_vertexes = RecdataProcess.from_18_to_42(xy_list)
        for i in range(4):
            center_x += four_vertexes[i][0] / 4
            center_y += four_vertexes[i][1] / 4

        return center_x, center_y

    @staticmethod
    def get_rec_shape_data(
        xy_list,
        center=True, 
        length_W=True, 
        length_H=True, 
        rotate_angle_W=True, 
        rotate_angle_H=True,
    ):
        """
        Parameters
        ----------
        xy_list：框四点坐标
        center：返回中心数据
        length_W：返回宽度数据
        length_H：返回长度数据
        rotate_angle_W：返回宽度向量旋转角数据
        rotate_angle_H：返回长度向量旋转角数据
        Returns
        ----------
        rec_shape_data：rec data字典
        """
        xy_list = RecdataProcess.reorder_rec(xy_list)
        rec_shape_data = {}


        if center:
            center_x, center_y = Recdata.get_center(xy_list)
            rec_shape_data['center'] = [center_x, center_y]

        if length_W or length_H or rotate_angle_H or rotate_angle_W:
            W1, W2, H1, H2 = Recdata.get_four_edge_vectors(xy_list)
        
        if length_W:
            length_W1, length_W2 = _get_vector_length(W1), _get_vector_length(W2)
            rec_shape_data['length_W'] = [length_W1, length_W2]

        if length_H:
            length_H1, length_H2 = _get_vector_length(H1), _get_vector_length(H2)
            rec_shape_data['length_H'] = [length_H1, length_H2]

        if rotate_angle_W:
            rotate_angle_W1, rotate_angle_W2 = (
                _get_vector_rotate_angle(W1), 
                _get_vector_rotate_angle(W2)
            )
            rec_shape_data['rotate_angle_W'] = [rotate_angle_W1, rotate_angle_W2]

        if rotate_angle_H:
            rotate_angle_H1, rotate_angle_H2 = (
                _get_vector_rotate_angle(H1), 
                _get_vector_rotate_angle(H2)
            )
            rec_shape_data['rotate_angle_H'] = [rotate_angle_H1, rotate_angle_H2]

        return rec_shape_data

    @staticmethod
    def get_draw_pos(xy_list, direction='center'):
        """
        获取基于每个rec框位置的绘制坐标
        Parameters
        ----------
        xy_list：框四点坐标
        # TODO：direction方位，可选'top'/'bottom'/'left'/'right'/'center'

        Returns
        ----------
        pos：绘制坐标
        """
        pass

    # TODO：尝试改进，例如设四点坐标，通过方程求解
    @staticmethod
    def get_xy_list(rec_shape_data):
        """
        从完整的rec_shape_data得到rec的四点坐标，以中心点为基准，结合高度、宽度向量进行运算
        Parameters
        ----------
        rec_shape_data：rec_shape的数据字典，包括center, length_W, length_H, 
        rotate_angle_W, rotate_angle_H

        Returns
        ----------
        xy_list：rec的四点坐标
        """
        _ = rec_shape_data
        center = _['center']
        length_W, length_H = _['length_W'], _['length_H']
        # ra = rotate_angle
        ra_W, ra_H = _['rotate_angle_W'], _['rotate_angle_H']
        vector_W1, vector_W2 = (
            Recdata.get_edge_vector(length_W[0], ra_W[0], is_edge_H=False),
            Recdata.get_edge_vector(length_W[1], ra_W[1], is_edge_H=False),

        )
        vector_H1, vector_H2 = (
            Recdata.get_edge_vector(length_H[0], ra_H[0], is_edge_H=True),
            Recdata.get_edge_vector(length_H[1], ra_H[1], is_edge_H=True),
        )
        # l/r/t/b = left/right/top/bottom
        # TODO：四条边的比例可以根据长度来变动
        rt = np.array(center) + 0.5 * np.array(vector_W1) + 0.5 * np.array(vector_H2)
        lt = np.array(center) - 0.5 * np.array(vector_W1) + 0.5 * np.array(vector_H1)
        rb = np.array(center) + 0.5 * np.array(vector_W2) - 0.5 * np.array(vector_H2)
        lb = np.array(center) - 0.5 * np.array(vector_W2) - 0.5 * np.array(vector_H1)
        # TODO：注意numpy返回四个点格式
        # TODO：从左上顶点逆时针
        return [rt[0], rt[1], lt[0], lt[1], lb[0], lb[1], rb[0], rb[1]]


class RecdataProcess(object):
    """
    对rec数据进行处理，包括重排点顺序，统一边长度
    """

    # TODO:重复的reshape可能有性能损失，暂不考虑
    @staticmethod
    def from_18_to_42(xy_list):
        """
        将1行8列array或list转为4行2列array，

        Parameters
        ----------
        xy_list: 1x8格式rec端点坐标

        Returns
        ----------
        reorder_xy_list:4x2格式rec端点坐标
        """
        xy_list = np.array(xy_list).reshape((4, 2))

        return xy_list

    @staticmethod
    def from_42_to_18(xy_list, return_list=False):
        """
        将4行2列array转为1行8列array或list，

        Parameters
        ----------
        xy_list: 1x8格式rec端点坐标

        Returns
        ----------
        reorder_xy_list:4x2格式rec端点坐标        
        """
        xy_list = np.array(xy_list).reshape(8)
        if return_list:
            xy_list = xy_list.tolist()

        return xy_list

    @staticmethod
    def reorder_vertexes(xy_list):
        """
        重排端点顺序，先找最小x作为起始端点，
        在剩下三点找中间y作为第三点，剩下两点按逆时针排序
        调整点顺序，使得13连线斜率大于24连线斜率

        Parameters
        ----------
        xy_list: rec的4端点坐标

        Returns
        ----------
        reorder_xy_list:重排后4x2格式rec端点坐标   
        """
        xy_list = RecdataProcess.from_18_to_42(xy_list)
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
            k[i] = (
                (xy_list[index, 1] - xy_list[first_v, 1])
                / (xy_list[index, 0] - xy_list[first_v, 0] + 1e-4)
            )
                        
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

    @staticmethod
    def reorder_rec(xy_list):
        """
        重排rec端点顺序
        在reorder vertexes的基础上，以第4点作为起点

        Parameters
        ----------
        xy_list: rec的4端点坐标

        Returns
        ----------
        reorder_xy_list:重排后4x2格式rec端点坐标        
        """
        xy_list = RecdataProcess.reorder_vertexes(xy_list)
        # 调整起点
        tmp_x, tmp_y = xy_list[3, 0], xy_list[3, 1]
        for i in range(3, 0, -1):
            xy_list[i] = xy_list[i - 1]
        xy_list[0, 0], xy_list[0, 1] = tmp_x, tmp_y
        reorder_rec_xy_list = np.reshape(xy_list, (1, 8))[0]

        # TODO: 原函数同时返回'test'
        return reorder_rec_xy_list

    @staticmethod
    def uniform_rec_edge_length(edge1, edge2):
        """
        将两边长度统一为较长边

        Parameters
        ----------
        edge1: rec的一边
        edge2: rec的一边
        edge1和edge2应为一组对边

        Returns
        ----------
        uniform_edge1: 统一后一边
        uniform_edge2: 统一后另一边
        """ 
        length1, length2 = np.linalg.norm(edge1), np.linalg.norm(edge2)
        if length1 > length2:
            k = length1 / length2
            uniform_edge1 = edge1
            uniform_edge2 = (k * edge2[0], k * edge2[1])
        else:
            k = length2 / length1
            uniform_edge1 = (k * edge1[0], k * edge1[1])
            uniform_edge2 = edge2

        return uniform_edge1, uniform_edge2

    @staticmethod
    def _shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio):

        if ratio == 0.0:
            return
        start_point = edge
        end_point = (edge + 1) % 4
        
        long_start_sign_x = np.sign(xy_list[end_point, 0] - xy_list[start_point, 0])
        new_xy_list[start_point, 0] = (
            xy_list[start_point, 0] + 
            long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
        )
            
        long_start_sign_y = np.sign(xy_list[end_point, 1] - xy_list[start_point, 1])
        new_xy_list[start_point, 1] = (
            xy_list[start_point, 1] + 
            long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
        )
            
        # long edge one, end point
        long_end_sign_x = -1 * long_start_sign_x
        new_xy_list[end_point, 0] = (
            xy_list[end_point, 0] +
            long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
        )
        long_end_sign_y = -1 * long_start_sign_y
        new_xy_list[end_point, 1] = (
            xy_list[end_point, 1] +
            long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])   
        )

    # TODO：研究代码
    @staticmethod
    def shrink(xy_list, ratio):    
        """
        收缩边
        Parameters
        ----------
    
        Returns
        ----------
        """
        if ratio == 0.0:
            return xy_list, xy_list
        # TODO：距离计算利用已有api
        # 四个向量，分别为V21, V32, V43, V14 
        diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
        diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
        diff = np.concatenate((diff_1to3, diff_4), axis=0)
        # 四向量长度
        dis = np.sqrt(np.sum(np.square(diff), axis=-1))

        # 选择最长边
        '''
        重排为[V1  V2
              V3   V4]
        沿axis = 0 相加，也就是选出最长的一组对边
        事实上xy_list已经经过重排，长边所在组已经固定
        '''
        # long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
        long_edge = 1
        short_edge = 1 - long_edge
        # 领边中的短边
        r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
        # cal theta array
        diff_abs = np.abs(diff)
        # 避免0
        diff_abs[:, 0] += EPSILON
        theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])

        # shrink two long edges
        temp_new_xy_list = np.copy(xy_list)
        RecdataProcess._shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
        RecdataProcess._shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)                            
        # shrink two short edges
        new_xy_list = np.copy(temp_new_xy_list)
        RecdataProcess._shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
        RecdataProcess._shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)

        return temp_new_xy_list, new_xy_list, long_edge      