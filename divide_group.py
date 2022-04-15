# -*- coding: utf-8 -*-
"""
Created on 2022-04-14 09:52:54

@author: Li Zhi
"""
from PIL import Image

from pkgs.recdata import recdata_correcting, recdata_processing, recdata_io
from pkgs.tool import visualization

PCA = recdata_correcting.PCA
Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess
RecdataIO = recdata_io.RecdataIO
RecDraw = visualization.RecDraw

# 不读取plate
def label2np(img_filepath, label_filepath, pca=False):
    recs_list = RecdataIO.read_rec_txt(label_filepath)
    recs_list = RecdataProcess.reorder_recs(recs_list)
    img = Image.open(img_filepath)
    w, h = img.size
    x_list, y_list = [], []
    if not pca:
        for rec in recs_list:
            if rec.classes == 'terminal':
                x, y = Recdata.get_center(rec.xy_list)
                x_list.append(x / w)
                y_list.append(y / h)
    else:
        recs_xy_list = []
        for rec in recs_list:
            if rec.classes == 'terminal':
                xy_list = []
                _ = rec.xy_list
                for i in range(4):
                    xy_list.append(_[2 * i] / w)
                    xy_list.append(_[2 * i + 1] / h)
                recs_xy_list.append(xy_list)
        pca_values = PCA.get_pca_values(recs_xy_list, preprocessing_='min_max')
        for i, pca_value in enumerate(pca_values):
            x_list.append(i)
            y_list.append(pca_value)
    return x_list, y_list


if __name__ == '__main__':

    x_list_list, y_list_list = [], []
    # 单副多列
    img_filepath_1 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/img/terminal_4.jpg'
    label_filepath_1 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/txt/terminal_4.txt'
    # 单副单列
    img_filepath_2 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/img/terminal_28.jpg'
    label_filepath_2 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/txt/terminal_28.txt'
    # 多副单列
    img_filepath_3 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/img/terminal_26.jpg'
    label_filepath_3 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/txt/terminal_26.txt'
    # 多个安装单位
    img_filepath_4 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/img/terminal_9.png'
    label_filepath_4 = 'D:/各种文件/图像识别/数据/端子排标注数据/标注整个边框/txt/terminal_9.txt'

    img = Image.open(img_filepath_4)
    recs_list = RecdataIO.read_rec_txt(label_filepath_4)
    recs_list = RecdataProcess.reorder_recs(recs_list)
    group_list = RecdataProcess.plate_group(recs_list)
    # group等于一个安装单位所属的端子
    group_num = 0
    for group in group_list:
        if group[0].classes == 'plate':
            RecDraw.draw_rec(group[0], img)
            # RecDraw.draw_text(f'第{group_num}组',group[0].xy_list, img)
            terminals_list = group[1:]
        else:
            terminals_list = group
        terminals_xy_list = [terminal.xy_list for terminal in terminals_list]
        pca_values = PCA.get_pca_values(terminals_xy_list)
        divide_groups = PCA.divide_recs(pca_values)
        index_groups = divide_groups['index']
        for index_group in index_groups:
            for rec_index in index_group:
                rec = terminals_list[rec_index]
                RecDraw.draw_rec(rec, img)
                RecDraw.draw_text(f'第{group_num}组', rec.xy_list, img)
            group_num += 1
    img.show()

