# -*- coding: utf-8 -*-
"""
Created on 2021-10-26 21:40:38

@author: Li Zhi
"""
import math
import os
import os.path as path
import random

import cv2
import numpy as np
from PIL import Image
from shapely import geometry

from . import cfg
from ..recdata import recdata_processing

EPSILON = 1e-4
Polygon = geometry.Polygon
RecdataProcess = recdata_processing.RecdataProcess

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
    @staticmethod
    def crop_rec(img, xy_list, coef_x_len=cfg.coef_x_len, coef_y_len=cfg.coef_y_len):
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
        # TODO：注意xy_list顺序
        W1, W2, H1, H2 = recdata_processing.Recdata.get_four_edge_vectors(xy_list)
        length_ = lambda vector: np.linalg.norm(vector)  #pylint: disable=W0108
        degree_ = lambda vector: math.degrees(math.atan(vector[1] / vector[0] + EPSILON))
        projection_length_ = lambda weight_sin, weight_cos, degree: (
            int(weight_sin * abs(math.sin(math.radians(degree)))) +
            int(weight_cos * abs(math.cos(math.radians(degree))))
        )
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

        # rt = right top, lb = left bottom
        xy_list_42 = RecdataProcess.from_18_to_42(RecdataProcess.reorder_rec(xy_list))
        # 注意考虑xy_list顺序
        rt, lb = list(xy_list_42[0]), list(xy_list_42[2])
        rt_new = np.dot(mat_rotation, np.array([rt[0], rt[1], 1]))
        lb_new = np.dot(mat_rotation, np.array([lb[0], lb[1], 1]))
        x_len, y_len = (
            int(coef_x_len * abs(rt_new[0] - lb_new[0])),
            int(coef_y_len * abs(rt_new[1] - lb_new[1])),
        )
        # TODO：限制范围避免超出，还要研究
        # 我觉得rt应该是min(x, 宽度),max(y, 1)；
        # TODO：一张更通用的方法求取端点，max min
        rt_new = [max(1, int(rt_new[0] + x_len)), max(1, int(rt_new[1] - y_len))]
        lb_new = [
            min(width_new - 1, int(lb_new[0] - x_len)), min(height_new - 1, int(lb_new[1] + y_len))
        ]
        rec_box = [lb_new[0], rt_new[1], rt_new[0], lb_new[1]]
        img_rotation = np.uint8(img_rotation)
        img_rotation = Image.fromarray(img_rotation)
        img_rec = img_rotation.crop(rec_box)

        return img_rec

    @staticmethod
    def joint_rec(
        img, 
        recs_xy_list,
        max_joint_img_width=cfg.max_joint_img_width, 
        joint_img_height=cfg.joint_img_height, 
        img_rec_height=cfg.img_rec_height, 
        spacing=cfg.spacing,
        joint_img_dir=cfg.joint_img_dir,
        preprocess_img_rec=cfg.preprocess_img_rec,
    ):
        """
        拼接多个rec为一张或多张图片
        Parameters
        ----------
        max_joint_img_width：单张joint_img的最大宽度，超出则新建另一张
        joint_img_height：joint_img的高度
        img_rec_height：每张img_rec的高度，resize后粘贴
        spacing：两张img_rec之间的间隔
        joint_img_dir：输出joint_img路径

        Returns
        ----------
        """
        img_rec_list = []
        for xy_list  in recs_xy_list:
            img_rec = ImageProcess.crop_rec(img, xy_list)
            assert (not img_rec.width == 0) and (not img_rec.height == 0), (
                f'{xy_list}裁切图片的宽高应大于0'
            )
            img_rec = img_rec.resize(
                (int(img_rec.width * img_rec_height / img_rec.height), img_rec_height)
            )
            img_rec_list.append(img_rec)

        available_width = max_joint_img_width
        joint_img_cnt = 0
        joint_img_name = f'joint_img_{joint_img_cnt}.jpg'
        joint_img = Image.new('RGB', (max_joint_img_width, joint_img_height), 'white')
        paste_x, paste_y = 0, int((joint_img_height - img_rec_height) / 2)
        for img_rec in img_rec_list:
            # TODO：预处理
            if available_width < spacing + img_rec.width:
                joint_img_path = path.join(joint_img_dir, joint_img_name)
                joint_img.save(joint_img_path)
                joint_img_cnt += 1
                joint_img_name = f'joint_img_{joint_img_cnt}'
                joint_img = Image.new('RGB', (max_joint_img_width, joint_img_height), 'white')
                available_width, paste_x = max_joint_img_width, 0
        
            joint_img.paste(img_rec, (paste_x, paste_y))
            available_width -= spacing + img_rec.width
            paste_x += spacing + img_rec.width

    # TODO：函数拆分，整理代码
    # TODO：主要花费时间为crop和save，后续尝试优化
    # TODO：根据img大小自适应裁切次数
    @staticmethod
    def random_crop(img, label, output_dir, count, boundary_thres, boundary_ratio, label_keyword):
        """
        随机裁切图片，检查标签是否在裁切区域内
        Parameters
        ----------
        img：img路径或文件夹（11/18，目前仅支持裁切文件夹中图片）
        label：label txt路径或文件夹
        output_dir：输出路径
        count：裁切次数
        boundary_thres：若相交部分占label大于boundary_thres，就视为边界label
        boundary_ratio：若边界label / 内部label > boundary_ratio，就舍弃该次裁切
        label_keyword：label必须包含的关键字

        Returns
        ----------
        """
        def crop_img(img_filepath, label_filepath):
            
            # TODO：裁切区域的宽高需要大于一定阈值
            def from_label_get_crop_region(
                label_region, img_width, img_height, x_threshold=None, y_threshold=None,
            ):
                
                label_x = [label_region[2 * i] for i in range(4)]
                label_y = [label_region[2 * i + 1] for i in range(4)]
                # 包络label的矩形
                label_xmin = math.floor(min(label_x))
                label_xmax = min(math.floor(max(label_x) + 1), img_width)
                label_ymin = math.floor(min(label_y))
                label_ymax = min(math.floor(max(label_y) + 1), img_height)
                
                crop_region_xmin = random.randint(0, label_xmin)
                crop_region_xmax = random.randint(label_xmax, img_width)
                crop_region_ymin = random.randint(0, label_ymin)
                crop_region_ymax = random.randint(label_ymax, img_height)

                crop_region = [
                    crop_region_xmin, crop_region_ymin, crop_region_xmax, crop_region_ymax
                ]

                return crop_region

            def is_label_in_crop_img(label_region, crop_region):
                # label_region：8个数格式
                # crop_region：4个数格式（左上，右下）
                xmin, xmax = crop_region[0], crop_region[2]
                ymin, ymax = crop_region[1], crop_region[3]
                for i in range(4):
                    if xmin < label_region[2 * i] < xmax and ymin < label_region[2 * i + 1] < ymax:
                        continue
                    else:
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
                # TODO：label_line += '\n'，split取最后一个，包含了换行

                return label_line

            with open(label_filepath, encoding='utf-8') as f:
                label_lines = f.readlines()

            get_region = lambda label_line: [float(point) for point in label_line.split(',')[:8]]
            get_label = lambda label_line: label_line.split(',')[-1]
            is_keyword_label = (
                lambda label_line: 
                    True if label_keyword is None or label_keyword in get_label(label_line)
                    else False
            )
            # TODO：根据label筛选会导致漏标区域内其他类型label
            original_label_lines = label_lines.copy()
            label_lines = [line for line in label_lines if is_keyword_label(line)]
            if not label_lines:
                return

            i = 0
            img = Image.open(img_filepath)
            width, height = img.size[0], img.size[1]

            while i < count:
                
                random_label_line = random.choice(label_lines)
                random_label_region = get_region(random_label_line)
                crop_region = from_label_get_crop_region(random_label_region, width, height)
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
                # 三类标签：内部，边界，外部
                # 如果一张图片包含了太多边界标签，就舍去
                for label_line in original_label_lines:
                # for label_line in label_lines:
                    label_region, label = get_region(label_line), get_label(label_line)
                    if is_label_in_crop_img(label_region, crop_region):
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
                output_img_path = path.join(output_dir, output_name + '.jpg')
                output_label_path = path.join(output_dir, output_name + '.txt')
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
        for img_file in img_files:
            if get_filename(img_file) in label_names:
                samename_img_files.append(img_file)

        for img_file in samename_img_files:
            img_filepath = path.join(img_dir, img_file)
            label_file = get_filename(img_file) + '.txt'
            label_filepath = path.join(label_dir, label_file)
            crop_img(img_filepath, label_filepath)
