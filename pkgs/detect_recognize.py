# -*- coding: utf-8 -*-
"""
Created on 2021-12-26 15:33:39

@author: Li Zhi
端到端模块，顶层封装
"""
from os import path

from keras import backend
from PIL import Image

from .east import east_net
from .recdata import recdata_database, recdata_processing, recdata_io
from .tool import visualization

EastNet = east_net.EastNet
RESULT_IMG_PATH = './resource/tmp.jpg'  # 基于根目录运行入口文件
RecdataDB = recdata_database.RecdataDB
RecdataProcess = recdata_processing.RecdataProcess
RecdataRecognize = recdata_processing.RecdataRecognize
RecdataIO = recdata_io.RecdataIO
RecDraw = visualization.RecDraw


# TODO：terminal5_number1，dibision by zero
# TODO：terminal5_number2，list index out of range
# TODO：terminal9没有任何数据
# TODO：针对correct中的pca，当数据过少时不执行操作
class EndToEnd(object):
    """
    端到端，实现检测及识别，输出txt结果，绘制结果img
    """
    def __init__(self):
        # clear_session以避免图错误
        backend.clear_session()
        self.east = EastNet()

    def load_weights(self):
        self.east.load_weights()

    def get_graph(self):
        """
        获取当前默认图，多线程相关
        Parameters
        ----------
        Returns
        ----------
        """
        graph = self.east.get_graph()
        return graph

    def detect_recognize(self, cubicle_id, img_path, test_rec_txt_path=None, *loops_num):
        """
        输入单张图片路径，完成检测及识别
        返回绘制识别结果的图片
        可以输出识别结果到txt，通过EastNet.predict实现
        不设置额外参数，实现细节参数通过各模块cfg文件设置默认值
        Parameters
        ----------
        cubicle_id：规范化后的计量柜id，扫描二维码后得到
        img_path：图片路径
        test_rec_txt_path：仅用于测试，不通过east识别，而是直接读取txt文件作为检测识别结果
        loops_num：回路编号

        Returns
        ----------
        result_img：图片，PIL.Image
        """
        img = Image.open(img_path)
        img_name = path.basename(img_path)
        db = RecdataDB(cubicle_id)

        if test_rec_txt_path is None:
            recs_list = self.east.predict(img_dir_or_path=img_path)
        else:
            recs_list = RecdataIO.read_rec_txt(test_rec_txt_path)
        recognize_recs_list = RecdataRecognize.recognize(img, img_name, recs_list) # 识别成功的
        group_list = RecdataProcess.plate_group(recognize_recs_list)
        group_list = db.get_terminals_id(group_list) # 生成端子id
        # 查询连接回路
        # for group in group_list:
        #     if not group[0].classes == 'plate':
        #         pass
        #         # 显示手动输入
        #     for i, rec in enumerate(group):
        #         if i == 0:
        #             continue
        #         terminal_id = rec.id_
        #         loops_id = db.get_connected_loops_id(terminal_id)
        #         for loop_id in loops_id:
        #             loop_num = db.get_loop_num(loop_id)
        #             if loop_num in loops_num:
        #                 print(f'{rec.xy_list}是待检端子')
