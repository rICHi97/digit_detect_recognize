# -*- coding: utf-8 -*-
"""
Created on 2022-04-13 10:15:39

@author: Li Zhi
"""
from PIL import Image
from matplotlib import markers
import  matplotlib.pyplot as plt

from pkgs.recdata import recdata_correcting,recdata_processing, recdata_io

PCA = recdata_correcting.PCA
Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess
RecdataIO = recdata_io.RecdataIO

rc = {
    "font.family" : "Times New Roman",
    "mathtext.fontset" : "stix",
}
label_font = {
    'family': 'SimSun',
    'size': 10.5,
}
arrowprops = {
    'arrowstyle': '->',
    'connectionstyle': 'arc3,rad=-.2',
    'linestyle': 'dashed',
}

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

def np2plot(x_list_list, y_list_list, pca=False):  #pylint: disable=W0621
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['lines.markersize'] = 6
    plt.tick_params(direction='in') # 刻度朝里
    if not pca:
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.xlim(0, 25)
        plt.xticks(range(0, 25, 2))
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
    # 我真的醉了，xticks就不可以用fontdict，xlabel就可以。设置刻度字体
    plt.xticks(fontfamily='Times New Roman', fontsize=10.5)
    plt.yticks(fontfamily='Times New Roman', fontsize=10.5)

    my_markers = ['+', 'o', 'x']
    handles = []
    for x_list, y_list, marker in zip(x_list_list, y_list_list, my_markers):
        if len(x_list) < 20:
            p = plt.scatter(x_list, y_list, s=36, marker=marker, color='black', linewidths=1)
        # 均匀奇偶采样
        # 0, 5, 10, 15
        else:
            step = len(x_list) // 20
            step = step + 1 if step % 2 == 0 else step
            if not pca:
                p = plt.scatter(x_list[::step], y_list[::step], s=36, marker=marker, color='black', linewidths=1)
            else:
                p = plt.scatter(range(0, len(x_list[:19 * step:step])), y_list[:19 * step:step], s=36, marker=marker, color='black', linewidths=1)
        handles.append(p)

    # 图例
    plt.legend(
        handles=handles,
        labels=['单副多列', '单副单列', '多副单列'],
        loc='upper right',
        prop=label_font,
        frameon=False,
    )

    # 轴名称
    if not pca:
        plt.xlabel(r'中心$x$坐标', fontdict=label_font)
        plt.ylabel(r'中心$y$坐标', fontdict=label_font)
    else:
        plt.xlabel(r'端子序号', fontdict=label_font)
        plt.ylabel(r'端子$\mathrm{PCA}$坐标', fontdict=label_font)

    plt.show()

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

    for img_filepath, label_filepath in zip(
        (img_filepath_1, img_filepath_2, img_filepath_3),
        (label_filepath_1, label_filepath_2, label_filepath_3),
    ):
        x_list, y_list = label2np(img_filepath, label_filepath, pca=True)
        x_list_list.append(x_list)
        y_list_list.append(y_list)


    np2plot(x_list_list, y_list_list, pca=True)

    dan_duo = y_list_list[0]
    dan_dan = y_list_list[1]
    duo_dan = y_list_list[2]

    step = len(y_list_list[0]) // 20
    step = step + 1 if step % 2 == 0 else step
    dan_duo = dan_duo[:20 * step:step]
