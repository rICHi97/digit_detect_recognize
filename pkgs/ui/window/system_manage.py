# -*- coding: utf-8 -*-
"""
Created on 2022-02-28 18:08:07

@author: Li Zhi
"""
from PyQt5 import QtWidgets

from ..qt_designer_code import Ui_system_manage_main_window

QMainWindow = QtWidgets.QMainWindow


class MainWindow(object):
    """
    系统管理主窗口
    """
    Ui_MainWindow = Ui_system_manage_main_window.Ui_MainWindow

    def __init__(self):
        self._init_ui()
        self._init_content()
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_main_window = self.Ui_MainWindow()

    # 实体
    def _init_content(self):
        self.main_window = QMainWindow()

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_main_window.setupUi(self.main_window)

    # 一个window的signal仅能处理本窗口中的事务
    # 如果需要涉及别的窗口，将其信号封装，在ui_integrated中连接
    # MainWindow().connect['clicked_connect_db'](callable_object)
    def _setup_signal(self):
        self.connect = {}

    def show(self):
        self.main_window.show()

    def update_label(self, username_pwd):
        text = username_pwd[0] + username_pwd[1]
        self.ui_main_window.label.setText(text)