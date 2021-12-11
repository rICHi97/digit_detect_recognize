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
import base64
import math
import requests

import numpy as np

from . import cfg

_api_key = '7j3KnKhBfvL5M46GwGIIOCBB'
_secret_key = 'OLjSdoILVVRMiKza088n4RFpWZXd5OKK'
_digit_request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/numbers'
_character_request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic'
_host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={_api_key}&client_secret={_secret_key}'
# TODO：token会变化吗？
_access_token = None
EPSILON = 1e-4

def _get_vector(length, rotate_angle):

    vector = [length * math.cos(rotate_angle), length * math.sin(rotate_angle)]

    return vector

def _get_vector_length(vector):

    length = np.linalg.norm(vector)

    return length

def _get_vector_rotate_angle(vector):

    rotate_angle = math.atan(vector[1] / (vector[0] + EPSILON))

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
        is_rotate_angle：待求取的rec_data是否为旋转角，只针对旋转角分别按正负求平均

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
    def get_four_edge_vectors(xy_list, return_array=False):
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
        xy_list = RecdataProcess.reorder_rec(xy_list)
        xy_list = RecdataProcess.from_18_to_42(xy_list)
        W1 = [xy_list[0][0] - xy_list[1][0], xy_list[0][1] - xy_list[1][1]]
        W2 = [xy_list[3][0] - xy_list[2][0], xy_list[3][1] - xy_list[2][1]]
        H1 = [xy_list[1][0] - xy_list[2][0], xy_list[1][1] - xy_list[2][1]]
        H2 = [xy_list[0][0] - xy_list[3][0], xy_list[0][1] - xy_list[3][1]]
        edge_vectors = [W1, W2, H1, H2]
        if return_array:
            edge_vectors = [np.array(V) for V in edge_vectors]

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
        xy_list：rec的四端点坐标

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
    def get_edge_vector_center(xy_list, edge_vector):
        """
        Parameters
        ----------
        xy_list：rec的四端点坐标
        edge_vetor：'W1'/'W2'/'H1'/'H2'

        Returns
        ----------
        vector_center：（x，y）向量的中心
        """
        xy_list = RecdataProcess.reorder_rec(xy_list)
        xy_list = RecdataProcess.from_18_to_42(xy_list)
        vector_centers = {
            'W1': (
                0.5 * xy_list[0][0] + 0.5 * xy_list[1][0], 0.5 * xy_list[0][1] + 0.5 * xy_list[1][1]
            ),
            'W2': (
                0.5 * xy_list[3][0] + 0.5 * xy_list[2][0], 0.5 * xy_list[3][1] + 0.5 * xy_list[2][1]
            ),
            'H1': (
                0.5 * xy_list[1][0] + 0.5 * xy_list[2][0], 0.5 * xy_list[1][1] + 0.5 * xy_list[2][1]
            ),
            'H2': (
                0.5 * xy_list[0][0] + 0.5 * xy_list[3][0], 0.5 * xy_list[0][1] + 0.5 * xy_list[3][1]
            ),

        }
        vector_center = vector_centers[edge_vector]

        return vector_center
        

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
        获取用于矫正的形状位置数据
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

    # TODO：尝试控制中心，不控制四向量权重
    @staticmethod
    def get_text_area(
        xy_list, 
        W_coef=cfg.W_coef, 
        H_coef=cfg.H_coef,
    ):
        """
        求取端子中心的文本（编号或铭牌信息）坐标
        首先分别计算W的x坐标中点，然后文本中心的x坐标为：W_coef * W1 + (1 - W_coef) * W2
        H同理
        Parameters
        ----------
        xy_list：rec四点坐标
        W_coef：计算文本中心x坐标时，W的权重。中心x坐标=W_coef * W1中点 + （1 - W_coef） * W2中点
        H_coef：同W_coef，H的权重

        Returns
        ----------
        text_xy_list：文本四点坐标
        """
        # coef = 0.5时，为中点；coef = 1时，为point_1
        coef_point = lambda point_1, point_2, coef: coef * point_1 + (1 - coef) * point_2
        # TODO：控制了中心点，就不控制各向量权重。
        vertex = lambda center, W, H: center + 0.25 * W + 0.25 * H
        edge_center_xy = []
        for edge_vector, xy in zip(('W1', 'W2', 'H1', 'H2'), (0, 0, 1, 1)):
            center_xy = Recdata.get_edge_vector_center(xy_list, edge_vector)[xy]
            edge_center_xy.append(center_xy)
        text_center_x = coef_point(edge_center_xy[0], edge_center_xy[1], W_coef)
        text_center_y = coef_point(edge_center_xy[2], edge_center_xy[3], H_coef)
        text_center = np.array([text_center_x, text_center_y])

        W1, W2, H1, H2 = Recdata.get_four_edge_vectors(xy_list, return_array=True)
        # r/l = right/left, t/b = top/bottom
        rt = vertex(text_center, +W1, +H2)  #pylint: disable=E1130
        lt = vertex(text_center, -W1, +H1)  #pylint: disable=E1130
        lb = vertex(text_center, -W2, -H1)  #pylint: disable=E1130
        rb = vertex(text_center, +W1, -H1)  #pylint: disable=E1130
        text_area = [rt[0], rt[1], lt[0], lt[1], lb[0], lb[1], rb[0], rb[1]]

        return text_area


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
        reorder_xy_list:np 4 * 2 array，rec端点坐标
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
        xy_list:1 * 8 rec端点坐标，array或list
        """
        xy_list = np.array(xy_list).reshape(8)
        if return_list:
            xy_list = xy_list.tolist()

        return xy_list
    
    @staticmethod
    def _reorder_vertexes(xy_list):
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
    
    # 应该使用reorder_rec来重排rec端点
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
        xy_list = RecdataProcess._reorder_vertexes(xy_list)
        # 调整起点
        tmp_x, tmp_y = xy_list[3, 0], xy_list[3, 1]
        for i in range(3, 0, -1):
            xy_list[i] = xy_list[i - 1]
        xy_list[0, 0], xy_list[0, 1] = tmp_x, tmp_y
        reorder_rec_xy_list = np.reshape(xy_list, (1, 8))[0]

        return reorder_rec_xy_list

    # TODO：单纯统一长度似乎没有必要
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


class RecdataRecognize(object):


    @staticmethod
    def _get_access_token():
        global _access_token, _host  #pylint: disable=W0603
        if _access_token is None:
            response = requests.get(_host)
            _access_token = response.json()['access_token']

        return _access_token

    @staticmethod
    def _request_post():
        ...

    @staticmethod
    def character_recognize(img_path):
        with open(img_path, 'rb') as f:
            img = base64.b64encode(f.read())
            params = {'image': img}
            access_token = RecdataRecognize._get_access_token()
            request_url = f'{_character_request_url}?access_token={access_token}'
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            response = requests.post(request_url, data=params, headers=headers)
            if response:
                print(response.json())

    # TODO：优化速度，joint_rec中的图片和数组操作可能是性能瓶颈
    # TODO：多个rec的排序问题
    @staticmethod
    def recognize(img, img_name, recs_xy_list, recs_classes_list):
        from . import recdata_correcting  #pylint: disable=C0415
        from ..tool import image_processing  #pylint: disable=C0415
        ImageProcess = image_processing.ImageProcess

        Correction = recdata_correcting.Correction
        recs_classes_set = set(recs_classes_list)
        for classes in recs_classes_set:
            recs_same_classes = []
            for i in range(len(recs_classes_list)):
                if recs_classes_list[i] == classes:
                    recs_same_classes.append(RecdataProcess.reorder_rec(recs_xy_list[i]))
            # TODO：裁切过程中好像裁切到了错误的rec，测试图片terminal_5_number_1.jpg
            if classes == '编号':
                # 矫正
                # TODO：非常紧急！correct后端子坐标好像有错
                corrected_recs_shape_data = Correction.correct_rec(recs_same_classes)
                corrected_recs_xy_list = [
                    Recdata.get_xy_list(rec_shape_data)
                    for rec_shape_data in corrected_recs_shape_data
                ]
                # 裁切拼接图片
                ImageProcess.joint_rec(img, img_name, corrected_recs_xy_list)
                # 调用数字识别

            elif classes == '铭牌':
                # 单独裁切每个rec，调用文字识别
                pass