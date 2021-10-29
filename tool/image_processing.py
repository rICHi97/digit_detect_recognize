# -*- coding: utf-8 -*-
"""
Created on 2021-10-26 21:40:38

@author: Li Zhi
"""
import math

import cv2
import numpy as np
from PIL import Image

import recdata_processing

EPSILON = 1e-4


class ImageProcess(object):
    """
    用于处理图片
    """
    coef_x_len, coef_y_len = 0.23, 0.02
    @staticmethod
    def dump_rotate_img(img, xy_list):
        """
        根据rec四点坐标，旋转图片从而裁切一个平置矩形
        Parameters
        ----------
        img：rec对应端子排图片,PIL图片格式
        xy_list：框四点坐标

        Returns
        ----------
        rec_img：裁切的端子图片，经过旋转
        """

        # 选择较长的一组对边中最大的倾斜角作为rec倾斜角
        W1, W2, H1, H2 = recdata_processing.Recdata.get_four_edge_vectors(xy_list)
        # gvl = get_vector_length, gvd = get_vector_degree, gpl = get_projection_length
        gvl = lambda vector: np.linalg.norm(vector)
        gvd = lambda vector: math.degrees(math.atan(vector[1] / vector[0] + EPSILON))
        gpl = lambda weight_sin, weight_cos, degree: (
            int(weight_sin * abs(math.sin(math.radians(degree)))) +
            int(weight_cos * abs(math.cos(math.radians(degree))))
        )   
        if max(gvl(W1), gvl(W2)) > max(gvl(H1), gvl(H2)):
            degree = gvd(W1) if abs(gvd(W1)) > abs(gvd(W2)) else gvd(W2)
        else:
            degree = 90 - abs(gvd(V1)) if abs(gvd(V1)) > abs(gvd(V2)) else 90 - abs(gvd(V2))
            sign = np.sign(gvd(V1)) if abs(gvd(V1)) > abs(gvd(V2)) else np.sign(gvd(V2))
            degree = -1 * sign * degree
        height, width = img.shape[:2]
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        
        
        height_new, width_new = gpl(width, height, degree), gpl(height, width, degree)
        # 拓展宽高
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2
        img_rotation = cv2.warpAffine(
            img, mat_rotation, (width_new, height_new), borderValue=(255, 255, 255),
        )

        # rt = right top, lb = left bottom
        xy_list_42 = recdata_processing.RecdataProcess.from_18_to_42(xy_list)
        rt, lb = list(xy_list_42[0]), list(xy_list_42[2])
        rt_new = np.dot(mat_rotation, np.array([rt[0], rt[1], 1]))
        lb_new = np.dot(mat_rotation, np.array([lb[0], lb[1], 1]))
        x_len, y_len = (
            int(coef_x_len * abs(rt_new[0] - lb_new[0])),
            int(coef_y_len * abs(rt_new[1] - lb_new[1])),
        )
        # TODO：限制范围避免超出，还要研究
        # 我觉得rt应该是min(x, 宽度),max(y, 1)；
        rt_new = [max(1, int(rt_new[0] + x_len)), max(1, int(rt_new[1] - y_len))]
        lb_new = [
            min(width_new - 1, int(lb_new[0] - x_len)), min(height_new - 1, int(lb_new[1] + y_len))
        ]
        rec_box = [lb_new[0], rt_new[1], rt_new[0], lb_new[1]]

        img_rotation = Image.fromarray(img_rotation)
        img_rec = img_rotation.crop(rec_box)

        return img_rec
