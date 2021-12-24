# -*- coding: utf-8 -*-
"""
Created on 2021-12-13 15:28:08

@author: Li Zhi
"""
import sys

from PyQt5 import QtWidgets

import connect_database_window
import choose_image_window
import main_window

QApplication = QtWidgets.QApplication

ConnectDBWindow = connect_database_window.ConnectDBWindow
ChooseIMGWindow = choose_image_window.ChooseIMGWindow
MainWindow = main_window.MainWindow

# TODO：子窗口位置跟随父窗口，直接给dialog设置父窗口好像显示有问题
class MainApp():

    def __init__(self):
        self._init_ui()
        self._setup_signal()

    def  _init_ui(self):
        self.connect_db_window = ConnectDBWindow()
        self.choose_img_window = ChooseIMGWindow()
        self.main_window = MainWindow()

    def _setup_signal(self):
        self.main_window.connect['click_connect_db'](
            self.connect_db_window.show
        )
        self.main_window.connect['click_choose_img'](
            self.choose_img_window.show
        )
        # self.connect_db_window['click_confirm_btn'](
        # )

    def show(self):  #pylint: disable=C0116
        self.main_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
