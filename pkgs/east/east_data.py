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

from keras import applications, callbacks, preprocessing
import numpy as np
from PIL import Image, ImageDraw
import tqdm
from tensorflow.compat import v1

from . import cfg
from ..recdata import recdata_processing

preprocess_input = applications.vgg16.preprocess_input
EarlyStopping = callbacks.EarlyStopping
LearningRateScheduler = callbacks.LearningRateScheduler
ModelCheckpoint = callbacks.ModelCheckpoint
ReduceLROnPlateau = callbacks.ReduceLROnPlateau
TensorBoard = callbacks.TensorBoard
tqdm = tqdm.tqdm
cast = v1.cast
equal = v1.equal
reduce_mean = v1.reduce_mean
reduce_sum = v1.reduce_sum
sigmoid = v1.nn.sigmoid

Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess

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
    # 训练时是固定size img
    # predict时resize_img
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
        # 判断似乎多余，如果等于max_img_size，就说明max_img_size <= img.width
        # 若为最大尺寸，则调整height，但调整后height不一定满足要求，要二次调整
        if img_width == max_img_size:
            ratio = img_width / img.width
            img_height = int(ratio * img.height)
        else:
            img_height = img.height

        # 按宽度比例调整图片后，高度不一定符合要求，第二次调整
        o_height = min(img_height, max_img_size)
        if o_height == max_img_size:
            ratio = o_height / img_height
            o_width = int(ratio * img_width)
        else:                             
            o_width = img_width
        d_width, d_height = o_width - (o_width % 32), o_height - (o_height % 32)

        return d_width, d_height

    # TODO：可以使用更小的图片进行训练，识别文字时按比例放大结果后裁切以加快速度
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
        verbose=cfg.preprocess_verbose,
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
        # 训练时固定size img, max_size * max_size
        # predict时resize, 保持图片原始大小，当原始大小大于max_size时按比例resize
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

            # 8位表示4点坐标，1位表示端子编号或铭牌
            recs_xy_list = np.zeros((len(annotation_list), 9))
            for anno, i in zip(annotation_list, range(len(annotation_list))):
                anno_columns = anno.strip().split(',') # 单个rec
                anno_array = np.array(anno_columns)

                # TODO：函数拆分
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = RecdataProcess.reorder_rec(xy_list)
                four_points = xy_list.copy()
                xy_list = np.reshape(xy_list, -1).tolist()
                if anno_array[8] == 'number':
                    xy_list.insert(1, 0)
                elif anno_array[8] == 'plate':
                    xy_list.insert(1, 1)
                recs_xy_list[i] = np.array(xy_list)  # 训练数据为原始xy_list经过按比例收缩和reorder

                # TODO：画框&保存resize原始图片，需要画框时才收缩

                # TODO：返回的第一个对象是收缩一组对边后的xy_list
                if show_preprocess_img:
                    W_H_ratio = _get_W_H_ratio(four_points)
                    _, shrink_xy_list, _ = RecdataProcess.shrink(four_points, shrink_ratio)
                    shrink_1, _, long_edge = RecdataProcess.shrink(
                        four_points, shrink_side_ratio * W_H_ratio
                    )
                    # 原始框
                    draw.line(
                        [
                            tuple(four_points[0]), 
                            tuple(four_points[1]),
                            tuple(four_points[2]), 
                            tuple(four_points[3]),
                            tuple(four_points[0]),
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
                                tuple(four_points[vs[long_edge][q_th][0]]),
                                tuple(shrink_1[vs[long_edge][q_th][1]]),
                                tuple(shrink_1[vs[long_edge][q_th][2]]),
                                tuple(four_points[vs[long_edge][q_th][3]]),
                                tuple(four_points[vs[long_edge][q_th][4]]),
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
        verbose=cfg.preprocess_verbose,
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

            groung_truth = np.zeros((height // pixel_size, width // pixel_size, 8))
            recs_xy_list = np.load(path.join(train_label_dir, img_name[:-4] + '.npy'))   

            if show_label_img:
                with Image.open(os.path.join(train_img_dir, img_name)) as img:
                    draw = ImageDraw.Draw(img)  # 用于显示label结果

            for xy_list in recs_xy_list:
                
                class_ = xy_list[1]
                xy_list = np.delete(np.array(xy_list), 1).reshape((4, 2))
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
                            groung_truth[i, j, 1] = class_
                            line_width, line_color = 1, 'red'
                            # 如果在首，ith = 0；在尾，ith = 1
                            # TODO：紧急！检查long edge是否正确和vs对应
                            ith = _point_inside_head_or_tail(px, py, xy_list, shrink_1, long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            # 固定long_edge = 0
                            # vs = [[[1, 2], [3, 0]]]
                            # in range(2) = 是首尾像素
                            if ith in range(2):
                                groung_truth[i, j, 2] = 1
                                groung_truth[i, j, 3] = ith
                                # 如果属于首像素，就是和第0 1点坐标之差
                                # 如果属于尾像素，就是和第2 3点坐标之差
                                groung_truth[i, j, 4:6] = xy_list[vs[long_edge][ith][0]] - [px, py]
                                groung_truth[i, j, 6:8] = xy_list[vs[long_edge][ith][1]] - [px, py]
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
    # region是x方向连接的多个像素，如果region与ij相连，should_merge = True
    neighbor = {(i, j - 1)}
    # 

    return not region.isdisjoint(neighbor)

def _initial_region_group(region_list):
    # 先对region进行初步分组
    region_groups = []
    region_group = []
    for region in region_list:
        i_index = region.copy().pop()[0] # copy以避免改变集合中的值，集合可变对象是传地址
        if not region_group:
            region_group.append(region)
            max_i_index = i_index
            continue
        if i_index - max_i_index < 2:
            region_group.append(region)
        else:
            region_groups.append(region_group)
            region_group = [region]
        max_i_index = i_index
    region_groups.append(region_group)

    return region_groups

def _region_group(region_list):
    # 1/4优化，先对region_list分组
    # 每组中region是基于i的邻近region，有可能是连通的（还需要判断j）
    # 不同组region之间不可能连通,只在每组region中计算连通region    
    initial_region_group = _initial_region_group(region_list)
    cnt_region_in_group = [len(_) for _ in initial_region_group]
    D, S_list, pre_cnt = [], [], 0
    for cnt in cnt_region_in_group:
        S_list.append(list(range(pre_cnt, pre_cnt + cnt)))
        pre_cnt += cnt
    # S中是region序号
    for S in S_list:
        while len(S) > 0:
            m = S.pop(0)
            if len(S) == 0:
                D.append([m])
            else:
                D.append(_rec_region_merge(region_list, m, S)) 

    return D

def _rec_region_merge(region_list, m, S):
    # 对于region_m，找到所有和它上下相连的region组tmp
    # 对于tmp中每个region，递归调用
    # 最后的rows是多组region，每组中region连通
    # a不一定与b相连，但可能通过a c d b之类的连通
    rows = [m]
    # tmp中为与region_m连通的region
    tmp = []
    # TODO：可能的优化方向:排序。不需要和S中所有n计算，只计算上方一行和下方一行中的n
    # S中传入的是region序号，不包括region所在行号。可能需要结合region_list计算，或更改输入参数
    # region_group耗时约20ms，可以暂不优化
    for n in S:
        # region_n在region_m下方相连或region_m在region_n下方相连
        # region_list[m]是m集合的地址，对region_list[m]改变会改变region_list中的值
        if (
            not _region_neighbor(region_list[m]).isdisjoint(region_list[n]) or
            not _region_neighbor(region_list[n]).isdisjoint(region_list[m])
        ):
            # 第m与n相交
            tmp.append(n)
    # 会改变S中内容
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(_rec_region_merge(region_list, e, S))
    return rows

def _region_neighbor(region_set):
    # 相连指的是两个region间至少有一个像素连通
    # e.g.
    #     * * r r r * *
    #     * n n n n n *
    # 其中*代表一般像素，r代表region中的像素，n代表region的neighbor
    # 1/4优化，使用list操作代替np 
    this_region_list = list(region_set)
    j_list = [ij[1] for ij in this_region_list]
    j_min, j_max = min(j_list) - 1, max(j_list) + 1
    i_m = this_region_list[0][0] + 1
    neighbor = {(i_m, ij[1]) for ij in this_region_list}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    
    return neighbor

class EastData(object):

    early_stopping_patience = 6
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

    # TODO：预测classes功能可能需要拆分
    # TODO：进一步研究nms
    @staticmethod
    def nms(
        predict,
        activation_pixels,
        side_vertex_pixel_threshold=cfg.side_vertex_pixel_threshold,
        trunc_threshold=cfg.trunc_threshold,
        pixel_size=cfg.pixel_size,
        return_classes=False,
    ):
        """
        对预测结果非最大抑制。
        Parameters
        ----------
        predict：Keras.Model.predict返回array的元素
        activation_pixels：predict中大于threshold的像素
        side_vertext_pixel_threshold：边界像素阈值
        trunc_threshold：头尾像素阈值

        Returns
        ----------
        score_list，rec_list：nms后多个rec四点得分list，nms后多个rec四点坐标
        """
        region_list = []
        # 形成多行region_list
        # TODO：下面的方法依赖np输出activation_pixels顺序，可能需要确认
        # 1/4优化，np输出activation_pixels按先行后列顺序给出
        # 即第一行的所有像素，第二行的所有像素；每行像素按列递增给出
        # x方向相连像素属于一个region，需要merge；
        # 像素不需要判断不同行的region，假设同一行有多个region，只需要和最后一个region判断
        # 综上，只需要和region_list中最后一个region判断即可（之前是和所有region）
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            merge = False
            if region_list:
                region = region_list[-1]
                if _should_merge(region, i, j):
                    region.add((i, j))
                    merge = True

            if not merge:
                region_list.append({(i, j)})

        D = _region_group(region_list)
        recs_xy_list = np.zeros((len(D), 4, 2))
        score_list = np.zeros((len(D), 4))
        recs_classes_list = []
        # 对于每个rec
        for group, g_th in zip(D, range(len(D))):
            total_score = np.zeros((4, 2))  
            classes_score, cnt = 0, 0
            for row in group:
                for ij in region_list[row]:
                    classes_score += predict[ij[0], ij[1], 1]
                    cnt += 1
                    # 边界得分
                    score = predict[ij[0], ij[1], 2]
                    # 如果是边界像素
                    if score >= side_vertex_pixel_threshold:
                        # ith_score表示头尾得分
                        ith_score = predict[ij[0], ij[1], 3]
                        if not trunc_threshold <= ith_score < 1 - trunc_threshold:
                            ith = int(np.around(ith_score))
                            # score为是否为边界像素得分
                            total_score[ith * 2:(ith + 1) * 2] += score
                            # TODO：recs_xy_list * cfg.pixel_size替代，原始是px,py * pixel_size
                            px, py = ij[1] + 0.5, ij[0] + 0.5
                            p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 4:8], (2, 2))
                            # 第g_th个rec的首或尾加上该像素得分*该像素判断首尾坐标
                            recs_xy_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
            classes_score /= cnt
            # TODO：设为函数参数，是否考虑加一类不确定
            recs_classes_list.append('编号' if classes_score < 0.5 else '铭牌')
            score_list[g_th] = total_score[:, 0]
            # recs_xy_list = (第i个像素score * 第i个像素p_v) / total_score
            recs_xy_list[g_th] = recs_xy_list[g_th] * pixel_size / (total_score + cfg.epsilon)

        if return_classes:
            return score_list, recs_xy_list, recs_classes_list
        return score_list, recs_xy_list
        
    @staticmethod
    def generator(
        batch_size=cfg.batch_size,
        data_dir=cfg.data_dir,
        is_val=False,
        train_img_dir=cfg.train_img_dir,
        train_label_dir=cfg.train_label_dir,
        train_filename=cfg.train_filename,
        val_filename=cfg.val_filename,
        num_channels=cfg.num_channels,
        max_train_img_size=cfg.max_train_img_size,
        pixel_size=cfg.pixel_size,
    ):
        """
        Parameters
        ----------
        
        Returns
        ----------
        """
        img_h, img_w = max_train_img_size, max_train_img_size
        x = np.zeros((batch_size, img_h, img_w, num_channels), dtype=np.float32)
        pixel_num_h = img_h // pixel_size   # fixme 以4*4个像素点为一格 预测值score
        pixel_num_w = img_w // pixel_size
        # 增加一维class score
        y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 8), dtype=np.float32)        
        # y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)
        if is_val:
            with open(os.path.join(data_dir, val_filename), 'r') as f_val:
                f_list = f_val.readlines()
        else:
            with open(os.path.join(data_dir, train_filename), 'r') as f_train:
                f_list = f_train.readlines()
        while True:
            for i in range(batch_size):
                # random gen an image name
                random_img = np.random.choice(f_list)
                # strip移除头尾字符 默认是空格或换行符
                img_filename = str(random_img).strip().split(',')[0]
                # load img and img anno
                img_path = os.path.join(data_dir, train_img_dir, img_filename)
                img = preprocessing.image.load_img(img_path)
                img = preprocessing.image.img_to_array(img)
                x[i] = preprocess_input(img, mode='tf')
                gt_file = os.path.join(data_dir, train_label_dir, img_filename[:-4] + '_gt.npy')
                y[i] = np.load(gt_file)
            yield x, y
    
    @staticmethod
    def rec_loss(
        y_true,
        y_pred,
        epsilon=cfg.epsilon,
        lambda_class_score_loss=cfg.lambda_class_score_loss,
        lambda_inside_score_loss=cfg.lambda_inside_score_loss,
        lambda_side_vertex_code_loss=cfg.lambda_side_vertex_code_loss,
        lambda_side_vertex_coord_loss=cfg.lambda_side_vertex_coord_loss,
    ):
        """
        Parameters
        ----------
       
        Returns
        ----------
        """
        # L2 distance
        def quad_norm(g_true):
            shape = v1.shape(g_true)
            delta_xy_matrix = v1.reshape(g_true, [-1, 2, 2])
            diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
            square = v1.math.square(diff)
            distance = v1.math.sqrt(reduce_sum(square, axis=-1))
            distance *= 4.0
            distance += epsilon

            return v1.reshape(distance, shape[:-1])

        def smooth_l1_loss(prediction_tensor, target_tensor, weights):
            n_q = v1.reshape(quad_norm(target_tensor), v1.shape(weights))
            diff = prediction_tensor - target_tensor
            abs_diff = v1.math.abs(diff)
            abs_diff_lt_1 = v1.math.less(abs_diff, 1)
            # （abs_didd - 0.5） 使用了L1距离计算方式 曼哈顿距离计算
            pixel_wise_smooth_l1norm = (
                reduce_sum(
                    v1.where(abs_diff_lt_1, 0.5 * v1.math.square(abs_diff), abs_diff - 0.5), 
                    axis=-1,
                ) / n_q * weights
            )

            return pixel_wise_smooth_l1norm
        # TODO：这个内部非边界像素有什么作用呢
        # inside 0：是否在内部
        # class 1：是编号还是铭牌
        # vertex_code 2：是否在边界
        # vertex_code 3：在首部还是尾部
        # vertex_coord 4-7：首部或尾部两端点坐标差

        # loss for inside_score
        inside_logits = y_pred[:, :, :, 0]
        inside_labels = y_true[:, :, :, 0]
        inside_predicts = sigmoid(inside_logits)
        inside_beta = 1 - reduce_mean(inside_labels)
        # log + epsilon for stable cal
        inside_score_loss = reduce_mean(
            -1 * (inside_beta * inside_labels * v1.math.log(inside_predicts + epsilon) +
            (1 - inside_beta) * (1 - inside_labels) * v1.math.log(1 - inside_predicts + epsilon))
        )
        inside_score_loss *= lambda_inside_score_loss

        # loss for class_score
        class_logits = y_pred[:, :, :, 1]
        class_labels = y_true[:, :, :, 1]
        class_predicts = sigmoid(class_logits)
        class_beta = 1 - reduce_mean(class_labels)
        class_score_loss = reduce_mean(
            -1 * (class_beta * class_labels * v1.math.log(class_predicts + epsilon) +
            (1 - class_beta) * (1 - class_labels) * v1.math.log(1 - class_predicts + epsilon))
        )
        class_score_loss *= lambda_class_score_loss

        # loss for side_vertex_code
        # fixme class-balanced cross-entropy 处理正负样本不均衡问题
        vertex_logits = y_pred[:, :, :, 2:4]
        vertex_labels = y_true[:, :, :, 2:4]
        # beta = 1 - 是否为边界/是否为内部
        vertex_beta = (
            1 - (reduce_mean(y_true[:, :, :, 2:3]) / (reduce_mean(inside_labels) + epsilon))
        )
        vertex_predicts = sigmoid(vertex_logits)
        pos = -1 * vertex_beta * vertex_labels * v1.math.log(vertex_predicts + epsilon)
        neg = -1 * (
            (1 - vertex_beta) * (1 - vertex_labels) * v1.math.log(1 - vertex_predicts + epsilon)
        )
        # cast：向下取整
        # positive_weights返回所有内部像素为1的tensor
        positive_weights = cast(v1.equal(y_true[:, :, :, 0], 1), v1.dtypes.float32)
        side_vertex_code_loss = (
            reduce_sum(reduce_sum(pos + neg, axis=-1) * positive_weights) /
            (reduce_sum(positive_weights) + epsilon)
        )
        side_vertex_code_loss *= lambda_side_vertex_code_loss

        # loss for side_vertex_coord delta
        # fixme 所有的边界像素预测值的加权平均来预测头或尾的短边两端的两个顶点
        # smoothed L1 loss
        g_hat = y_pred[:, :, :, 4:]
        g_true = y_true[:, :, :, 4:]
        # veryex_weights返回所有边界像素为1的tensor
        vertex_weights = cast(v1.equal(y_true[:, :, :, 2], 1), v1.dtypes.float32)
        pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
        side_vertex_coord_loss = (
            reduce_sum(pixel_wise_smooth_l1norm) /
            (reduce_sum(vertex_weights) + epsilon)
        )
        side_vertex_coord_loss *= lambda_side_vertex_coord_loss

        return inside_score_loss + class_score_loss + side_vertex_code_loss + side_vertex_coord_loss

    @staticmethod
    def callbacks(type_=None,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_verbose=cfg.early_stopping_verbose,
        check_point_filepath=cfg.check_point_filepath,
        check_point_period=cfg.check_point_period,
        check_point_verbose=cfg.check_point_verbose,
        reduce_lr_monitor=cfg.reduce_lr_monitor,
        reduce_lr_factor=cfg.reduce_lr_factor,
        reduce_lr_patience=cfg.reduce_lr_patience,
        reduce_lr_verbose=cfg.reduce_lr_verbose,
        reduce_lr_min_lr=cfg.reduce_lr_min_lr,
        # TODO：tensorboard
    ):
        """
        Parameters
        ----------  
        Returns
        ----------
        """
        if type_ not in ['early_stopping', 'check_point', 'reduce_lr']:
            return
        if type_ == 'early_stopping':
            callback = EarlyStopping(patience=early_stopping_patience, verbose=early_stopping_verbose)
        elif type_ == 'check_point':
            # TODO：默认monitor
            callback = ModelCheckpoint(
                filepath=check_point_filepath, 
                save_best_only=True,
                save_weights_only=True,
                period=check_point_period,
                verbose=check_point_verbose,
            )
        elif type_ == 'reduce_lr':
            callback = ReduceLROnPlateau(
                monitor=reduce_lr_monitor,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                verbose=reduce_lr_verbose,
                min_lr=reduce_lr_min_lr,
            )

        return callback