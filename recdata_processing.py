# -*- coding: utf-8 -*-
"""
Created on 2021-09-06 18:41:29

@author: Li Zhi
"""
import numpy as np


class RecData(object):
    """
    从rec四点坐标得到rec data，如边长度、中心坐标
    """

    @staticmethod
    def get_four_edges(xy_list):
        """
        得到四边向量元组
        返回顺序为：W1，W2，H1，H2
        Parameters
        ----------
        xy_list: rec的4端点坐标

        Returns
        ----------
        four_edges: 四边向量，按W1 W2 H1 H2顺序
        """
        four_vertexes = np.array(xy_list).reshape(4, 2)
        W1 = (four_vertexes[0][0] - four_vertexes[1][0], four_vertexes[0][1] - four_vertexes[1][1])
        W2 = (four_vertexes[3][0] - four_vertexes[2][0], four_vertexes[3][1] - four_vertexes[2][1])
        H1 = (four_vertexes[1][0] - four_vertexes[2][0], four_vertexes[1][1] - four_vertexes[2][1])
        H2 = (four_vertexes[0][0] - four_vertexes[3][0], four_vertexes[0][1] - four_vertexes[3][1])
        four_edges = [W1, W2, H1, H2]

        return four_edges

    @staticmethod
    def get_center(xy_list):
        """
        得到rec中心坐标
        Parameters
        ----------
        
        Returns
        ----------
        """
        
class RecProcessing(object):
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
        xy_list = RecProcessing.from_18_to_42(xy_list)
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
        xy_list = np.array(xy_list).reshape((4, 2))
        xy_list = reorder_vertexes(xy_list)
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
            uniform_edge2

        return uniform_edge1, uniform_edge2

