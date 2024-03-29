# -*- coding: utf-8 -*-
"""
Created on 2021-09-29 20:08:09

@author: Li Zhi
本模块用以实现数据的输入输出，包括从txt文件读取rec数据，将rec数据保存到txt等
一个rec的四个端点：xy_list
一张图片的多个rec的端点：recs_xy_list
多张图片的recs字典：imgs_rec_dict
"""
# TODO：将该模块移至tool中，实现更多数据的io
import fnmatch
import json
import os
import os.path as path

import numpy as np

from .rec import Rec

class RecdataIO(object):
    """
    读取txt文件，写入txt文件
    """
    @staticmethod
    def read_rec_txt(txt_path):
        """
        读取一张图片的rec txt文件
        将其转为该图片中所有rec四点坐标的列表
        Parameters
        ----------
        txt_path：rec txt路径

        Returns
        recs_list：多个rec实例
        ----------
        """
        recs_list = []
        with open(txt_path, 'r', encoding='utf8') as rec_txt:
            lines = rec_txt.readlines()
            for line in lines:
                line = line.strip().split(',')
                classes = None
                # len = 8，不含类别信息
                # len = 9，包含类别信息
                if len(line) == 9:
                    if line[-1] == 'number' or line[-1] == '编号':
                        classes = 'terminal'
                    else:
                        classes = 'plate'
                    line = line[:-1]
                xy_list = [float(xy) for xy in line]
                rec = Rec(xy_list, classes)
                recs_list.append(rec)

        return recs_list

    @staticmethod
    def read_rec_txt_dir(txt_dir, keyword=None):
        """
        对一个文件夹中的txt进行read_rec_txt操作
        Parameters
        ----------
        txt_dir：rec txt的文件夹
        keyword：如果不为None，则只读取文件名包含keyword的txt

        Returns
        ----------
        imgs_rec_dict：dict，键为txt文件名，值为该txt的recs_xy_list
        """
        imgs_rec_dict, txts = {}, []
        for file in os.listdir(txt_dir):
            if fnmatch.fnmatch(file, '*.txt'): # 匹配带有拓展名.txt
                txts.append(file)
            if keyword is not None:
                txts = [txt for txt in txts if keyword in txt]

        cwd = os.getcwd()
        os.chdir(txt_dir)
        for txt in txts:
            imgs_rec_dict[txt] = RecdataIO.read_rec_txt(txt)
        os.chdir(cwd)

        return imgs_rec_dict

    @staticmethod
    def read_rec_npy(npy_path):
        """
        输出结果基于resize后img
        Parameters
        ----------

        Returns
        ----------
        """
        npy = np.load(npy_path)
        recs_xy_list = npy[:, 0:8].tolist()
        recs_classes_list = npy[:, -1].tolist()
        recs_classes_list = ['编号' if classes < 0.5 else '铭牌' for classes in recs_classes_list]

        return recs_xy_list, recs_classes_list

    @staticmethod
    def read_gt(gt_path):
        """
        读取一个_gt.npy文件，返回recs_list和class信息
        Parameters
        ----------
        gt_path：gt文件路径

        Returns
        ----------
        xy_list和classes_list通gt文件获取，基于resize后img
        recs_xy_list：多个rec的四点坐标
        recs_classes_list：多个rec的类别信息
        """
        from ..east import east_data  #pylint: disable=C0415

        EastData = east_data.EastData

        predicts = np.load(gt_path)
        # gt，无需sigmoid
        activation_pixels = np.where(np.greater_equal(predicts[:, :, 0], 1))
        recs_score, recs_after_nms, classes_list = EastData.nms(
            predicts, activation_pixels, return_classes=True
        )

        recs_xy_list, recs_classes_list = [], []
        for score, xy_list, classes in zip(recs_score, recs_after_nms, classes_list):
            if np.amin(score) > 0:
                # TODO：classes分数设为参数，通过gt得到，应该仅为0或1
                xy_list = np.reshape(xy_list, (8,)).tolist()
                recs_xy_list.append(xy_list)
                recs_classes_list.append(classes)

        return recs_xy_list, recs_classes_list

    # TODO：cfg设置txt_dir
    @staticmethod
    def write_rec_txt(recs_xy_list, txt_dir, txt_name, recs_classes_list=None):
        """
        Parameters
        ----------
        recs_xy_list：多个rec的四点坐标
        txt_dir：输出txt文件夹
        txt_name：输出txt名字

        Returns
        ----------
        """
        lines = []
        for i, xy_list in enumerate(recs_xy_list):
            line = ','.join([f'{xy:.2f}' for xy in xy_list])
            if recs_classes_list is not None:
                line += f',{recs_classes_list[i]}'
            line += '\n'
            lines.append(line)
        txt_path = path.join(txt_dir, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as t:
            t.writelines(lines)

    # TODO：仅转换单个json文件
    @staticmethod
    def json_to_txt(json_dir, txt_dir=None, format_='LabelImage'):
        """
        将json格式标签转为txt
        Parameters
        ----------
        json_dir：json文件存放文件夹
        txt_dir：输出txt文件夹，若为None，则选择json_dir
        format_：json格式，'LabelImage'/'LabelMe'

        Returns
        ----------
        """
        assert format_ in ('LabelImage', 'LabelMe')

        if txt_dir is None:
            txt_dir = json_dir

        json_files = os.listdir(json_dir)
        for file in json_files:
            json_path = path.join(json_dir, file)
            with open(json_path, encoding='utf-8') as json_file:
                lines = []
                this_json = json.loads(json_file.read())

                if format_ == 'LabelImage':
                    for node in this_json:
                        line = ''
                        xy_list = node['content']
                        # TODO：label对应txt区分
                        if node['labels']['labelName'] == '未命名':
                            label = 'number'
                        else:
                            label = 'plate'
                        if not len(xy_list) == 4:
                            print(f'{json_path}错误，应为4点')
                            return
                        for point in xy_list:
                            line += '%.2f,%.2f,'%(point['x'], point['y'])
                        line += '%s\n'%(label)
                        lines.append(line)

                elif format_ == 'LabelMe':
                    for node in this_json:
                        shapes = node['shapes']

                        for shape in shapes:
                            line = ''
                            xy_list = []
                            for _ in shape['points']:
                                xy_list += _
                            xy_list = ','.join([f'{_:.2f}' for _ in xy_list])
                            label = shape['label']
                            line = f'{xy_list},{label}'

                            lines.append(line)

            txt_file = file[:-5]+ '.txt'
            txt_path = path.join(txt_dir, txt_file)
            with open(txt_path, 'w') as txt_file:
                txt_file.writelines(lines)

    # TODO：merge时需要结合json具体结构
    @staticmethod
    def merge_json(json1, json2, output_dir, json1_keyword=None, json2_keyword=None, indent=4):
        """
        合并两个json文件
        如果是两个文件夹，就合并同名json文件
        Parameters
        ----------
        json1：json文件路径或文件夹
        json2：json文件路径或文件夹
        output_dir：输出文件夹，不能与之前的重复；输出json会取json1的名字
        json1/2_keyword：仅合并包含关键字的键

        Returns
        ----------
        """
        is_keyword_in_label = (
            lambda node, keyword:
                True if keyword is None or keyword in node['labels']['labelName'] else False
        )

        def merge_two_json_file(json1_filepath, json2_filepath):

            with open(json1_filepath, encoding='utf-8') as json1_file:
                json1 = json.loads(json1_file.read())
            with open(json2_filepath, encoding='utf-8') as json2_file:
                json2 = json.loads(json2_file.read())

            merge_json1 = [node for node in json1 if is_keyword_in_label(node, json1_keyword)]
            merge_json2 = [node for node in json2 if is_keyword_in_label(node, json2_keyword)]
            merge_json = merge_json1 + merge_json2

            return merge_json

        json1_dir = json1 if path.isdir(json1) else path.dirname(json1)
        json2_dir = json2 if path.isdir(json2) else path.dirname(json2)
        # TODO：若输入json文件夹与output_dir相同，抛出异常
        # TODO：仅合并两个json文件

        # 合并两个文件夹中的同名json文件
        json1_files, json2_files = os.listdir(json1_dir), os.listdir(json2_dir)
        json_samename = [json_file for json_file in json1_files if json_file in json2_files]
        for file in json_samename:
            json1_filepath = path.join(json1_dir, file)
            json2_filepath = path.join(json2_dir, file)
            merge_json = merge_two_json_file(json1_filepath, json2_filepath)
            merge_json_path = path.join(output_dir, file)
            with open(merge_json_path, 'w', encoding='utf-8') as f:
                json.dump(merge_json, f, ensure_ascii=False, indent=indent)
