# -*- coding: utf-8 -*-
"""
Created on 2021-12-13 15:28:08

@author: Li Zhi
"""
from os import path
import sys

from PyQt5 import QtWidgets

from . import connect_database_window
from . import choose_image_window
from . import init_window
from . import main_window
from . import my_thread

QApplication = QtWidgets.QApplication

ConnectDBWindow = connect_database_window.ConnectDBWindow
ChooseIMGWindow = choose_image_window.ChooseIMGWindow
InitWindow = init_window.InitWindow
MainWindow = main_window.MainWindow
DetectRecognizeThread = my_thread.DetectRecognizeThread
LoadThread = my_thread.LoadThread

RESULT_IMG_PATH = './resource/tmp.jpg'

# TODO：注意temp图片，可能会调用之前识别结果，使用读入内存数据创建QImage
# TODO：选择terminal10时识别的为terminal9图片
# TODO：init dialog后设置parent好像会出问题，导致dialog不显示
# 但是在初始化时给定parent参数可以设置，子窗口会在父窗口的中心出现
class MainApp():

    def __init__(self):
        self.end_to_end = None
        self.graph = None
        self._init_ui()
        self._setup_signal()

    def _init_ui(self):
        self.init_window = InitWindow()
        self.main_window = MainWindow()
        self.choose_img_window = ChooseIMGWindow(self.main_window.main_window)
        self.connect_db_window = ConnectDBWindow(self.main_window.main_window)
        self.detect_recognize_thread = DetectRecognizeThread()

    def _setup_signal(self):
        self.init_window.connect['init_finished'](self.init_finished)
        self.main_window.connect['clicked_connect_db'](self.connect_db_window.show)
        self.main_window.connect['clicked_choose_img'](self.choose_img_window.show)
        self.main_window.connect['clicked_start_recognize'](self.start)
        self.choose_img_window.connect['finish_choose_img'](self.update_img)
        self.connect_db_window.connect['finish_connect_db'](self.update_db_status)
        self.detect_recognize_thread.finished.connect(self.detect_recognize_finished)

    def init_finished(self):
        """
        初始化完成
        Parameters
        ----------
        Returns
        ----------
        """
        self.end_to_end = self.init_window.end_to_end
        self.graph = self.init_window.graph
        self.main_window.show()

    def detect_recognize_finished(self):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        self.main_window.set_terminal_img(RESULT_IMG_PATH)
        self.main_window.ui_main_window.pushButton_3.setEnabled(True)

    def show(self):  #pylint: disable=C0116
        self.init_window.show()

    def start(self):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        if self.choose_img_window.img_path is None:
            # TODO：弹出msg_box提醒
            pass
        else:
            self.main_window.ui_main_window.pushButton_3.setEnabled(False)
            self.detect_recognize_thread.my_start(
                self.end_to_end, self.graph, self.choose_img_window.img_path
            )

    def update_img(self):
        """
        依据选择图片窗口所选择图片，更新label
        Parameters
        ----------
        Returns
        ----------
        """
        if self.choose_img_window.img_path is not None:
            self.main_window.set_terminal_img(self.choose_img_window.img_path)

    def update_db_status(self):
        """
        更新数据库状态
        Parameters
        ----------
        Returns
        ----------
        """
        if self.connect_db_window.db_path is not None:
            db_type = self.connect_db_window.db_type
            db_path = path.basename(self.connect_db_window.db_path)
            text = f'已连接到{db_type}数据库：{db_path}'
            self.main_window.set_db_status('connected', text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
