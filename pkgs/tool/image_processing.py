# -*- coding: utf-8 -*-
"""
Created on 2021-10-26 21:40:38

@author: Li Zhi
本模块对图片进行处理。
"""
import math
import os
from os import path
import random

import cv2
import numpy as np
from PIL import Image
from shapely import geometry

from . import cfg
from ..recdata import rec, recdata_processing

Polygon = geometry.Polygon

Rec = rec.Rec
Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess

EPSILON = 1e-4

def _get_img(img):

    # TODO：路径检查
    if isinstance(img, Image.Image):
        pass
    elif isinstance(img, str):
        img = Image.open(img)

    return img


# TODO：img参数类型应该统一
class ImageProcess(object):
    """
    用于处理图片
    """
    # TODO：仅裁切数字部分
    # TODO:尝试使用透视变换替代当前的旋转变换
    @staticmethod
    def crop_rec(
        img, xy_list, coef_x_len=cfg.coef_x_len, coef_y_len=cfg.coef_y_len
    ):
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
        img = _get_img(img).convert('RGB')
        img = np.asarray(img, 'f')
        img.flags.writeable = True
        # TODO：degree_用弧度表示，projection_lenth_可以去掉转弧度过程
        length_ = lambda vector: np.linalg.norm(vector)  #pylint: disable=W0108
        degree_ = lambda vector: math.degrees(math.atan(vector[1] / vector[0] + EPSILON))
        projection_length_ = lambda weight_sin, weight_cos, degree: (
            int(weight_sin * abs(math.sin(math.radians(degree)))) +
            int(weight_cos * abs(math.cos(math.radians(degree))))
        )

        # xy_list = Recdata.get_text_area(xy_list)
        # TODO：得到端子中心的文本部分
        W1, W2, H1, H2 = Recdata.get_four_edge_vectors(xy_list)
        if max(length_(W1), length_(W2)) > max(length_(H1), length_(H2)):
            degree = degree_(W1) if abs(degree_(W1)) > abs(degree_(W2)) else degree_(W2)
        else:
            V = H1 if abs(degree_(H1)) > abs(degree_(H2)) else H2
            degree = degree_(V)
            degree = -90 + degree if np.sign(degree) > 0 else 90 + degree
        height, width = img.shape[:2]
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  #pylint: disable=E1101
        height_new, width_new = (
            projection_length_(width, height, degree), projection_length_(height, width, degree)
        )
        # 拓展宽高
        # matr_rotation[0, 2] = delta_x, mat_rotation[1, 2] = delta_y
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2
        img_rotation = cv2.warpAffine(  #pylint: disable=E1101
            img, mat_rotation, (width_new, height_new), borderValue=(255, 255, 255),
        )
        img_rotation = np.uint8(img_rotation)
        img_rotation = Image.fromarray(img_rotation)

        # rt = right top, lb = left bottom
        xy_list = RecdataProcess.from_18_to_42(xy_list)
        rt, lb = list(xy_list[0]), list(xy_list[2])
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
        img_rec = img_rotation.crop(rec_box).convert('L')

        return img_rec

    # TODO：裁切铭牌无效
    # 2/4，传递数据使用recs_list，之前的似乎有问题，每个plate直接return，计数cnt始终等于0
    @staticmethod
    def joint_rec(
        img,
        img_name,
        recs_list,
        max_joint_img_width=cfg.max_joint_img_width,
        joint_img_height=cfg.joint_img_height,
        img_rec_height=cfg.img_rec_height,
        spacing=cfg.spacing,
        joint_img_dir=cfg.joint_img_dir,
    ):
        """
        拼接多个rec为一张或多张图片
        对于一张图片中的所有recs，如果为plate，就直接裁切；如果为terminal，就拼接
        Parameters
        ----------
        img：PIL.Image或img_path
        img_name：不含拓展名的图片名
        max_joint_img_width：单张joint_img的最大宽度，超出则新建另一张
        joint_img_height：joint_img的高度
        img_rec_height：每张img_rec的高度，resize后粘贴
        spacing：两张img_rec之间的间隔
        joint_img_dir：输出joint_img路径
        --joint_img_dir：所有joint_img的dir
        ----this_joint_img_dir:该张端子排图片所有classes rec存储在一个dir中

        Returns
        ----------
        joint_data：字典，结构为
        {'classes_cnt.jpg':[Rec1, Rec2, Rec3]}
        """
        # TODO：铭牌大小不能超过api限制
        this_joint_img_dir = path.join(joint_img_dir, img_name)
        if not path.exists(this_joint_img_dir):
            os.mkdir(this_joint_img_dir)

        terminal_cnt, plate_cnt = 0, 0
        joint_data = {}
        # 仅terminal需要joint
        joint_img = Image.new('RGB', (max_joint_img_width, joint_img_height), 'white')
        available_width = max_joint_img_width
        paste_x, paste_y = 0, int((joint_img_height - img_rec_height) / 2)

        for rec in recs_list:

            img_rec = ImageProcess.crop_rec(img, rec.xy_list)
            assert not img_rec.width * img_rec.height == 0, (f'{rec.xy_list}裁切宽高应大于0')

            if rec.classes == 'plate':
                img_rec.save(path.join(this_joint_img_dir, f'plate_{plate_cnt}.jpg'))
                joint_data[f'plate_{plate_cnt}.jpg'] = [rec]
                plate_cnt += 1

            elif rec.classes == 'terminal':
                # 仅terminal需要resize拼接
                img_rec = img_rec.resize(
                    (int(img_rec.width * img_rec_height / img_rec.height), img_rec_height)
                )
                # 当前背景图片可用空间不够，保存并新建
                if available_width < spacing + img_rec.width:
                    joint_img.save(path.join(this_joint_img_dir, f'terminal_{terminal_cnt}.jpg'))
                    terminal_cnt += 1
                    joint_img = Image.new('RGB', (max_joint_img_width, joint_img_height), 'white')
                    available_width = max_joint_img_width
                    paste_x = 0
                rec.set_attr(joint_x_position=(paste_x, paste_x + img_rec.width))
                joint_img.paste(img_rec, (paste_x, paste_y))
                paste_x += img_rec.width + spacing
                if f'terminal_{terminal_cnt}.jpg' not in joint_data.keys():
                    joint_data[f'terminal_{terminal_cnt}.jpg'] = [rec]
                else:
                    joint_data[f'terminal_{terminal_cnt}.jpg'].append(rec)

        joint_img.save(path.join(this_joint_img_dir, f'terminal_{terminal_cnt}.jpg'))

        return joint_data

    # TODO：可选预处理方式
    @staticmethod
    def preprocess_img(
        img,
        threshold=cfg.threshold,
    ):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        img_in = np.array(img)
        img_out = np.zeros(img_in.shape, np.uint8)
        # 归一化
        cv2.normalize(img_in, img_out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        if threshold:
            blur = cv2.GaussianBlur(img_out, (3, 3), 0)
            _, img_out = cv2.threshold(blur, 118, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU) # 返回：阈值，thres图片
        img_rec = Image.fromarray(img_out.astype(np.int32))

        return img_rec

    # TODO：函数拆分，整理代码
    # TODO：主要花费时间为crop和save，后续尝试优化
    # TODO：根据img大小自适应裁切次数
    # TODO：cfg参数化
    @staticmethod
    def random_crop(
        img,
        label,
        output_img_dir,
        output_label_dir,
        count,
        boundary_thres,
        boundary_ratio,
        label_keyword,
    ):
        """
        随机裁切图片，检查标签是否在裁切区域内
        选择图片短边边长，从中心随机裁切一个正方形，正方形边长在[min(短边边长, 100) -1, 短边边长]
        Parameters
        ----------
        img：img路径或文件夹（11/18，目前仅支持裁切文件夹中图片）
        label：label txt路径或文件夹
        output_dir：输出路径
        count：裁切次数/'auto'
        boundary_thres：若相交部分占label大于boundary_thres，就视为边界label
        boundary_ratio：若边界label / 内部label > boundary_ratio，就舍弃该次裁切
        label_keyword：label必须包含的关键字

        Returns
        ----------
        """
        def crop_img(img_filepath, label_filepath):

            def get_random_crop_region(width, height):

                max_crop_length = height if width > height else width
                # 随机选择裁切长度
                crop_length = (
                    max_crop_length if max_crop_length <= 150 else
                    random.choice(range(150, max_crop_length))
                )

                if width > height:
                    # 随机选择裁切中心，当宽大于高时，裁切区域沿宽度方向滑动，中心y坐标不变
                    c_x, c_y = (
                        random.choice(range(crop_length / 2, width - crop_length / 2)),
                        height / 2,
                    )
                else:
                    c_x, c_y = (
                        width / 2,
                        random.choice(range(crop_length / 2, width - crop_length / 2)),
                    )
                left = max(int(c_x - crop_length / 2), 0)
                right = min(int(c_x + crop_length /2), width)
                top = max(int(c_y - crop_length / 2), 0)
                bottom = min(int(c_y + crop_length / 2), height)

                crop_region = [left, top, right, bottom]

                return crop_region

            def is_label_in_crop_region(label_region, crop_region):
                # label_region：8个数格式
                # crop_region：4个数格式（左上，右下）
                xmin, xmax = crop_region[0], crop_region[2]
                ymin, ymax = crop_region[1], crop_region[3]
                for i in range(4):
                    if xmin < label_region[2 * i] < xmax and ymin < label_region[2 * i + 1] < ymax:
                        continue
                    return False
                return True

            def update_label_region(label_region, crop_region):

                xmin, ymin = crop_region[0], crop_region[1]
                new_label_region = []
                for i in range(4):
                    new_x = label_region[2 * i] - xmin
                    new_y = label_region[2 * i + 1] - ymin
                    new_label_region.append(new_x)
                    new_label_region.append(new_y)

                return new_label_region

            def get_label_line(label_region, label):

                label_region = [f'{point:.2f}' for point in label_region]
                label = str(label)
                label_region.append(label)
                label_line = ','.join(label_region)

                return label_line

            with open(label_filepath, encoding='utf-8') as f:
                label_lines = f.readlines()

            #从一行记录获取区域信息
            get_region = lambda label_line: [float(point) for point in label_line.split(',')[:8]]
            # 从一行记录获取label信息
            get_label = lambda label_line: label_line.split(',')[-1]
            # label是否包含关键词
            is_keyword_label = (
                lambda label_line:
                    bool(label_keyword is None or label_keyword in get_label(label_line))
            )

            original_label_lines = label_lines.copy()
            label_lines = [line for line in label_lines if is_keyword_label(line)]
            if not label_lines:
                return

            i = 0
            img = Image.open(img_filepath)
            width, height = img.size[0], img.size[1]
            if count == 'auto':
                short_edge_length = min(width, height)
                classes_coef = 10 if label_keyword == 'terminal' else 1
                base_count = 1 * (short_edge_length // 1000 +1)
                count = classes_coef * base_count

            while i < count:
                # 随机生成一个初步裁切区域
                crop_region = get_random_crop_region(width, height)
                _ = [
                        crop_region[0], crop_region[1],
                        crop_region[2], crop_region[1],
                        crop_region[2], crop_region[3],
                        crop_region[0], crop_region[3],
                    ]
                c = Polygon(np.array(_).reshape((4, 2)))
                crop_lines = []
                cnt_interior, cnt_boundary = 0, 0
                # TODO：需要遍历所有标签，可能需要优化
                # 三类标签：内部，边界，外部;如果一张图片包含了太多边界标签，就舍去
                for label_line in original_label_lines:
                    label_region, label = get_region(label_line), get_label(label_line)
                    # TODO：先缓存，因为后面这些数据有可能放弃
                    if is_label_in_crop_region(label_region, crop_region):
                        new_label_region = update_label_region(label_region, crop_region)
                        crop_line = get_label_line(new_label_region, label)
                        crop_lines.append(crop_line)
                        cnt_interior += 1
                    # 计算label有多少部分在裁切区域内
                    else:
                        l = Polygon(np.array(label_region).reshape((4, 2)))
                        inter = Polygon(c).intersection(Polygon(l)).area
                        if (inter / l.area) > boundary_thres:
                            cnt_boundary += 1
                if cnt_boundary > boundary_ratio * cnt_interior:
                    continue

                nonlocal img_file, get_filename
                filename = get_filename(img_file)
                output_name = f'{filename}_{label_keyword}_{i}'
                output_img_path = path.join(output_img_dir, output_name + '.jpg')
                output_label_path = path.join(output_label_dir, output_name + '.txt')
                img_crop = img.crop(crop_region)
                img_crop.save(output_img_path)
                with open(output_label_path, 'w', encoding='utf-8') as f:
                    f.writelines(crop_lines)

                i += 1

        # TODO：仅裁切一张图片和对应label
        img_dir = img if path.isdir(img) else path.dirname(img)
        label_dir = label if path.isdir(label) else path.dirname(label)
        img_files, label_files = os.listdir(img_dir), os.listdir(label_dir)
        get_filename = lambda file: file.split('.')[0]  # 去除格式拓展名
        label_names = [get_filename(label_file) for label_file in label_files]
        samename_img_files = []
        # 确定待裁切图片，仅裁切有对应label_txt的图片
        for img_file in img_files:
            if get_filename(img_file) in label_names:
                samename_img_files.append(img_file)

        for img_file in samename_img_files:
            img_filepath = path.join(img_dir, img_file)
            label_file = get_filename(img_file) + '.txt'
            label_filepath = path.join(label_dir, label_file)
            crop_img(img_filepath, label_filepath)

    @staticmethod
    def segment(img_path):

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thres, img_segment = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img_segment
