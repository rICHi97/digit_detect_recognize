# -*- coding: utf-8 -*-
"""
Created on 2021-11-10 01:00:14

@author: Li Zhi
"""

"""
本模块用以进行east网络的数据处理，包括预处理和产生标签，data_generator；
"""
import math
import os
from os import path
import random

import numpy as np
import tqdm
from PIL import Image, ImageDraw

from . import cfg
from ..recdata import recdata_processing

Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess
tqdm = tqdm.tqdm

def _get_W_H_ratio(xy_list):
    """
    检测的端子长宽接近，控制收缩比
    Parameters
    ----------
    xy_list：重排点顺序后端子四点坐标

    Returns
    ----------
    W_H_ratio：宽高比
    """
    rec_shape_data = Recdata.get_rec_shape_data(
        xy_list, 
        center=False,
        length_W=True,
        length_H=True,
        rotate_angle_H=False,
        rotate_angle_W=False,
    )
    W1, W2 = rec_shape_data['length_W']
    H1, H2 = rec_shape_data['length_H']
    W_H_ratio = (W1 + W2) / (H1 + H2)
    if W_H_ratio < 1:
        W_H_ratio = 2 * math.log(W_H_ratio + 1)

    return W_H_ratio

# TODO：p_min,p_max多余，就是shrink_xy_list
def _is_point_inside_shrink(px, py, shrink_xy_list, p_min, p_max):
    # 判断是否在内部像素外接矩形内
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = shrink_xy_list[1:4, :] - shrink_xy_list[:3, :]
        xy_list[3] = shrink_xy_list[0, :] - shrink_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        # 交换两列顺序
        yx_list[:, :] = shrink_xy_list[:, -1:-3:-1]
        # 矢量叉乘
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False

def _point_inside_head_or_tail(px, py, xy_list, shrink_1, long_edge):
    """
    判断在头部还是在尾部。
    通过两次收缩得到内部像素，shrink_1是收缩一次结果
    Parameters
    ----------
    
    Returns
    ----------
    """
    nth = -1
    # 点索引
    # TODO：更易读的写法
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
         [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]

    # 强制long_edge = 0
    # vs = [[[1, 1, 2, 2, 1], [3, 3, 0, 0, 3]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if _is_point_inside_shrink(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


class EastPreprocess(object):
    """
    本类用于预处理数据，包括预处理及生成标签
    """
    @staticmethod
    def resize_img(img, max_img_size=cfg.max_train_img_size):
        """

        Parameters
        ----------

        Returns
        ----------
        """
        img_width = min(img.width, max_img_size)
        # if img_width == max_img_size < img.width  起到and作用
        # TODO：判断似乎多余，如果等于max_img_size，就说明max_img_size <= img.width
        # 首先确定resize后width，为实际尺寸和最大允许尺寸中较小的
        # 若为最大尺寸，则调整height，但调整后height不一定满足要求，要二次调整
        if img_width == max_img_size:
            ratio = img_width / img.width
            img_height = int(ratio * img.height)
        else:
            img_height = img.height

        o_height = min(img_height, max_img_size)
        if o_height == max_img_size:
            ratio = o_height / img_height
            o_width = int(ratio * img_width)
        else:                             
            o_width = img_width
        d_width, d_height = o_width - (o_width % 32), o_height - (o_height % 32)

        return d_width, d_height

    # TODO：默认参数设置cfg
    @staticmethod
    def preprocess(
        data_dir=cfg.data_dir,
        origin_img_dir=cfg.origin_img_dir, 
        origin_txt_dir=cfg.origin_txt_dir, 
        train_img_dir=cfg.train_img_dir, 
        train_label_dir=cfg.train_label_dir,
        show_preprocess_img=cfg.show_preprocess_img,
        preprocess_img_dir=cfg.preprocess_img_dir,
        max_train_img_size=cfg.max_train_img_size,
        shrink_ratio=cfg.shrink_ratio,
        shrink_side_ratio=cfg.shrink_side_ratio,
        val_ratio=cfg.val_ratio,
        val_filename=cfg.val_filename,
        train_filename=cfg.train_filename,
        verbose=False,
    ):
        """
        统一端点顺序
        Parameters
        ----------
        路径示意：
        # data_dir:
        #     origin_img_dir;  原始图片
        #     origin_txt_dir;  原始txt
        #     train_img_dir;  训练用resize图片
        #     train_label_dir;  label文件，np/npy
        #     label_img_dir;  label后将小方格图片，标示首尾像素
        #     preprocess_img_dir;  preprocess后图片
        #     val_file;
        #     train_file;    
        # data_dir为绝对路径或相对路径，其余dir为字符串，相对data_dir
        show_preprocess：是否保存preprocess后小方格标签图片
        max_train_img_size：最大训练图片尺寸
        shrink_ratio：收缩比
        val_ratio：验证集比例
        val_filename/train_filename：验证集与训练集文件名称
        verbose：True，print额外信息；False，不print

        Returns
        ----------
        """
        # TODO：路径不存在就mkdir
        get_dir = lambda sub_dir: path.join(data_dir, sub_dir)
        origin_img_dir = get_dir(origin_img_dir)
        origin_txt_dir = get_dir(origin_txt_dir)
        train_img_dir = get_dir(train_img_dir)
        train_label_dir = get_dir(train_label_dir)
        preprocess_img_dir = get_dir(preprocess_img_dir)

        origin_img_list = os.listdir(origin_img_dir)
        if verbose:
            print(f'原始图片共{len(origin_img_list)}张。')

        train_val_set = []
        d_width, d_height = max_train_img_size, max_train_img_size
        for origin_img_name, _ in zip(origin_img_list, tqdm(range(len(origin_img_list)))):  #pylint: disable=E1102

            origin_img_path = path.join(origin_img_dir, origin_img_name)
            origin_txt_path = path.join(origin_txt_dir, origin_img_name[:-4] + '.txt')

            with Image.open(origin_img_path) as img:
                scale_ratio_w = d_width / img.width
                scale_ratio_h = d_height / img.height
                # resize后原始图片
                img = img.resize((d_width, d_height), Image.ANTIALIAS).convert('RGB')
                if show_preprocess_img:
                    preprocess_img = img.copy()
                    draw = ImageDraw.Draw(preprocess_img)  # gt图片用于标示label，与训练数据无关

            with open(origin_txt_path, 'r', encoding='utf-8') as f:
                annotation_list = f.readlines()  # 单张图片标注

            recs_xy_list = np.zeros((len(annotation_list), 4, 2))
            for anno, i in zip(annotation_list, range(len(annotation_list))):
                anno_columns = anno.strip().split(',') # 单个rec
                anno_array = np.array(anno_columns)

                # TODO：函数拆分
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = RecdataProcess.reorder_vertexes(xy_list)
                recs_xy_list[i] = xy_list  # 训练数据为原始xy_list经过按比例收缩和reorder

                # TODO：画框&保存resize原始图片，需要画框时才收缩

                # TODO：返回的第一个对象是收缩一组对边后的xy_list
                if show_preprocess_img:
                    W_H_ratio = _get_W_H_ratio(xy_list)
                    _, shrink_xy_list, _ = RecdataProcess.shrink(xy_list, shrink_ratio)
                    shrink_1, _, long_edge = RecdataProcess.shrink(
                        xy_list, shrink_side_ratio * W_H_ratio
                    )
                    # 原始框
                    draw.line(
                        [
                            tuple(xy_list[0]), 
                            tuple(xy_list[1]),
                            tuple(xy_list[2]), 
                            tuple(xy_list[3]),
                            tuple(xy_list[0]),
                        ],
                        width=1,
                        fill='green',
                    )
                    
                    # shrink后框
                    draw.line(
                        [
                            tuple(shrink_xy_list[0]),
                            tuple(shrink_xy_list[1]),
                            tuple(shrink_xy_list[2]),
                            tuple(shrink_xy_list[3]),
                            tuple(shrink_xy_list[0]),
                        ],
                        width=1,
                        fill='blue',
                    )
                    
                    # 应该是用来判断首尾边界的
                    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    # vs = [[[1, 1, 2, 2, 1], [3, 3, 0, 0, 3]]]
                    for q_th in range(2):
                        draw.line(
                            [
                                tuple(xy_list[vs[long_edge][q_th][0]]),
                                tuple(shrink_1[vs[long_edge][q_th][1]]),
                                tuple(shrink_1[vs[long_edge][q_th][2]]),
                                tuple(xy_list[vs[long_edge][q_th][3]]),
                                tuple(xy_list[vs[long_edge][q_th][4]]),
                            ],
                            width=1,
                            fill='yellow',
                        )

            train_img_path = path.join(train_img_dir, origin_img_name)
            img.save(train_img_path)
            train_label_path = path.join(train_label_dir, origin_img_name[:-4] + '.npy')
            np.save(train_label_path, recs_xy_list)
            if show_preprocess_img:
                preprocess_img_path = path.join(preprocess_img_dir, origin_img_name)
                preprocess_img.save(preprocess_img_path)
            train_val_set.append(f'{origin_img_name},{d_width},{d_height}\n')

        train_img_list, train_label_list = os.listdir(train_img_dir), os.listdir(train_label_dir)
        if verbose:
            print(f'训练图片共{len(train_img_list)}。\n训练标签共{len(train_label_list)}\n')

        random.shuffle(train_val_set)
        val_count = int(val_ratio * len(train_val_set))
        with open(path.join(data_dir, val_filename), 'w') as f:
            f.writelines(train_val_set[:val_count])
        with open(path.join(data_dir, train_filename), 'w') as f:
            f.writelines(train_val_set[val_count:])        

    @staticmethod
    def label(
        data_dir=cfg.data_dir,
        train_img_dir=cfg.train_img_dir, 
        train_label_dir=cfg.train_label_dir,
        show_label_img=cfg.show_label_img,
        label_img_dir=cfg.label_img_dir,
        shrink_ratio=cfg.shrink_ratio,
        shrink_side_ratio=cfg.shrink_side_ratio,
        pixel_size=cfg.pixel_size,
        val_filename=cfg.val_filename,
        train_filename=cfg.train_filename,
        verbose=False,
    ):
        """
        产生标签文件
        Parameters
        ----------
        
        Returns
        ----------
        """
        get_dir = lambda sub_dir: path.join(data_dir, sub_dir)
        train_label_dir = get_dir(train_label_dir)
        train_img_dir = get_dir(train_img_dir)
        label_img_dir = get_dir(label_img_dir)
        with open(path.join(data_dir, val_filename), 'r') as f_val:
            f_list = f_val.readlines()
        with open(path.join(data_dir, train_filename), 'r') as f_train:
            f_list.extend(f_train.readlines())

        for line, _ in zip(f_list, tqdm(range(len(f_list)))):  #pylint: disable=E1102
            # line为每张图片信息
            line_cols = str(line).strip().split(',')
            img_name, width, height = (
                line_cols[0].strip(),
                int(line_cols[1].strip()),
                int(line_cols[2].strip()),
            )

            groung_truth = np.zeros((height // pixel_size, width // pixel_size, 7))
            recs_xy_list = np.load(path.join(train_label_dir, img_name[:-4] + '.npy'))   

            if show_label_img:
                with Image.open(os.path.join(train_img_dir, img_name)) as img:
                    draw = ImageDraw.Draw(img)  # 用于显示label结果

            for xy_list in recs_xy_list:
                
                W_H_ratio = _get_W_H_ratio(xy_list)
                _, shrink_xy_list, _ = RecdataProcess.shrink(xy_list, shrink_ratio)
                shrink_1, _, long_edge = RecdataProcess.shrink(
                    xy_list, shrink_side_ratio * W_H_ratio
                )
                
                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / pixel_size - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / pixel_size - 0.5).astype(int) + 3
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(height // pixel_size, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(width // pixel_size, ji_max[0])
                for i in range(imin, imax):
                    for j in range(jmin, jmax):
                        px = (j + 0.5) * pixel_size
                        py = (i + 0.5) * pixel_size
                        if _is_point_inside_shrink(px, py, shrink_xy_list, p_min, p_max):
                            groung_truth[i, j, 0] = 1
                            line_width, line_color = 1, 'red'
                            # 如果在首，ith = 0；在尾，ith = 1
                            ith = _point_inside_head_or_tail(px, py, xy_list, shrink_1, long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            # 固定long_edge = 0
                            # vs = [[[1, 2], [3, 0]]]
                            # in range(2) = 是首尾像素
                            if ith in range(2):
                                groung_truth[i, j, 1] = 1
                                groung_truth[i, j, 2:3] = ith
                                # 如果属于首像素，就是和第0 1点坐标之差
                                # 如果属于尾像素，就是和第2 3点坐标之差
                                groung_truth[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                                groung_truth[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]
                                line_width = 2
                                line_color = 'yellow' if ith == 0 else  'green'

                            if show_label_img:
                                draw.line(
                                    [
                                        (px - 0.5 * pixel_size, py - 0.5 * pixel_size),
                                        (px + 0.5 * pixel_size, py - 0.5 * pixel_size),
                                        (px + 0.5 * pixel_size, py + 0.5 * pixel_size),
                                        (px - 0.5 * pixel_size, py + 0.5 * pixel_size),
                                        (px - 0.5 * pixel_size, py - 0.5 * pixel_size),
                                    ],
                                    width=line_width,
                                    fill=line_color,
                                )
                if show_label_img:
                    label_img_path = path.join(label_img_dir, img_name)
                    img.save(label_img_path)

            np.save(os.path.join(train_label_dir, img_name[:-4] + '_gt.npy'), groung_truth)

def _should_merge(region, i, j):

    neighbor = {(i, j - 1)}
    # 判断region是否包含有neighbor元素，不包含返回true

    return not region.isdisjoint(neighbor)   

def _region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor

def _region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(_rec_region_merge(region_list, m, S))
    return D

def _rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not _region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not _region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(_rec_region_merge(region_list, e, S))
    return rows

class EastData(object):

    @staticmethod
    def sigmoid(x):
        """
        Parameters:
        x：array

        Returns:
        y：y = 1 / (1 + exp(-x))
        """
        y = 1 / (1 + np.exp(-x))
        return y

    @staticmethod
    def nms(
        predict, 
        activation_pixels, 
        side_vertex_pixel_threshold=cfg.side_vertex_pixel_threshold,
        trunc_threshold=cfg.trunc_threshold,
        ):
        """
        Parameters
        ----------
        
        Returns
        ----------
        """
        region_list = []
        # 形成多行region_list
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            merge = False
            for region in enumerate(region_list):
                if _should_merge(region, i, j):
                    region.add((i, j))
                    merge = True

            if not merge:
                region_list.append({(i, j)})

        D = _region_group(region_list)
        quad_list = np.zeros((len(D), 4, 2))
        score_list = np.zeros((len(D), 4))
        for group, g_th in zip(D, range(len(D))):
            total_score = np.zeros((4, 2))
            for row in group:
                for ij in region_list[row]:
                    score = predict[ij[0], ij[1], 1]
                    # 如果是边界像素
                    if score >= side_vertex_pixel_threshold:
                        # ith_score表示头或尾
                        ith_score = predict[ij[0], ij[1], 2:3]
                        if not trunc_threshold <= ith_score < 1 - trunc_threshold:
                            # TODO：四舍五入ith=0/1表示头/尾
                            ith = int(np.around(ith_score))
                            # score为是否为边界像素得分
                            total_score[ith * 2 : (ith + 1) * 2] += score
                            px = (ij[1] + 0.5) * cfg.pixel_size
                            py = (ij[0] + 0.5) * cfg.pixel_size
                            p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                            quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
            score_list[g_th] = total_score[:, 0]
            quad_list[g_th] /= (total_score + cfg.epsilon)

        return score_list, quad_list
            

