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
from .. import detect_recognize
from . import main_window

QApplication = QtWidgets.QApplication

ConnectDBWindow = connect_database_window.ConnectDBWindow
ChooseIMGWindow = choose_image_window.ChooseIMGWindow
OUTPUT_IMG_PATH = detect_recognize.OUTPUT_IMG_PATH
EndToEnd = detect_recognize.EndToEnd
MainWindow = main_window.MainWindow

# TODO：init dialog后设置parent好像会出问题，导致dialog不显示
# 但是在初始化时给定parent参数可以设置，子窗口会在父窗口的中心出现
class MainApp():

    def __init__(self):
        self._init_ui()
        self._setup_signal()

    def  _init_ui(self):
        self.main_window = MainWindow()
        self.choose_img_window = ChooseIMGWindow(self.main_window.main_window)        
        self.connect_db_window = ConnectDBWindow(self.main_window.main_window)
        self.end_to_end = EndToEnd()

    def _setup_signal(self):
        self.main_window.connect['clicked_connect_db'](self.connect_db_window.show)
        self.main_window.connect['clicked_choose_img'](self.choose_img_window.show)
        self.main_window.connect['clicked_start_recognize'](self.start)
        self.choose_img_window.connect['finish_choose_img'](self.update_img)
        self.connect_db_window.connect['finish_connect_db'](self.update_db_status)

    def show(self):  #pylint: disable=C0116
        self.main_window.show()

    # TODO：多线程
    # TODO：可以检查已存的output_img是否为当前图片，避免反复识别
    def start(self):
        if self.choose_img_window.img_path is None:
            # TODO：弹出msg_box提醒
            pass
        else:
            self.end_to_end.detect_recognize(self.choose_img_window.img_path)
            self.main_window.set_terminal_img(OUTPUT_IMG_PATH)

    def update_img(self):
        """
        Parameters
        ----------
        
        Returns
        ----------
        """
        if self.choose_img_window.img_path is not None:
            self.main_window.set_terminal_img(self.choose_img_window.img_path)

    def update_db_status(self):
        """
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
