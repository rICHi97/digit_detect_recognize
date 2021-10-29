# -*- coding: utf-8 -*-
"""
Created on 2021-09-29 20:08:09

@author: Li Zhi
"""

"""
本模块用以实现数据的输入输出，包括从txt文件读取rec数据，将rec数据保存到txt等
一个rec的四个端点：xy_list
一张图片的多个rec的端点：recs_xy_list
多张图片的recs字典：imgs_rec_dict
"""
import fnmatch
import json
import os


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
        recs_xy_list：多个rec的四点坐标
        ----------
        """
        recs_xy_list = []
        with open(txt_path, 'r', encoding='utf8') as rec_txt:
            lines = rec_txt.readlines()
            for line in lines:
                line = line.split(',')
                xy_list = [float(xy) for xy in line]
                recs_xy_list.append(xy_list)

        return recs_xy_list

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
    def write_rec_txt(recs_xy_list, txt_dir, txt_name):
        """
        Parameters
        ----------
        
        Returns
        ----------
        """
        # TODO
        pass

    @staticmethod
    def json_to_txt(json_dir, txt_dir=None):
        """
        将json格式标签转为txt
        Parameters
        ----------
        json_dir：json文件存放文件夹
        txt_dir：输出txt文件夹，若为None，则选择json_dir
        
        Returns
        ----------
        """
        if txt_dir is None:
            txt_dir = json_dir

        json_files = os.listdir(json_dir)
        for file in json_files:
            json_path = os.path.join(json_dir, file)
            with open(json_path, encoding='utf-8') as json_file:
                lines = []
                this_json = json.loads(json_file.read())
                for node in this_json:
                    line = ''
                    xy_list = node['content']
                    if not len(xy_list) == 4:
                        print(f'{json_path}错误，应为4点')
                        return
                    for point in xy_list:
                        line += '%.2f,%.2f,'%(point['x'], point['y'])
                    line += 'number\n'
                    lines.append(line)

            txt_file = file[:-5]+ '.txt'
            txt_path = os.path.join(txt_dir, txt_file)
            with open(txt_path, 'w') as txt_file:
                txt_file.writelines(lines)