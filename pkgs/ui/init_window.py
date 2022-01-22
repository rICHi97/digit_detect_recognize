# -*- coding: utf-8 -*-
"""
Created on 2022-01-20 02:24:12

@author: Li Zhi
"""
from PyQt5 import QtCore, QtWidgets

from .import my_thread
from .qt_designer_code import Ui_init_window

QTimer = QtCore.QTimer
QDialog = QtWidgets.QDialog
LoadThread = my_thread.LoadThread
Ui_Dialog = Ui_init_window.Ui_Dialog


# TODO：位置变动后，主窗口跟随
class InitWindow():
    """
    初始化
    """
    def __init__(self):
        self._init_ui()
        self._init_content()
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_dialog = Ui_Dialog()

    # 实体
    def _init_content(self):
        self.dialog = QDialog()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.load_thread = LoadThread()
        self.end_to_end = None
        self.graph = None

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_dialog.setupUi(self.dialog)

    def _setup_signal(self):
        self.load_thread.finished.connect(self.update_status)
        self.timer.timeout.connect(self.dialog.close)
        self.connect = {
            'init_finished': self.timer.timeout.connect,
        }


    # TODO：除更新文本外也更新图片
    def update_status(self):
        """
        更新文本状态
        Parameters
        ----------
        Returns
        ----------
        """
        self.end_to_end = self.load_thread.end_to_end  #pylint: disable=W0201
        self.graph = self.load_thread.graph  #pylint: disable=W0201
        self.ui_dialog.label_3.setText(
            "<html><head/><body><p><font size=\"5\", face=\"黑体\">"
            "初始化完成.</font></p></body></html>"
        )
        self.timer.start(1500)

    def show(self):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        self.dialog.show()
        self.load_thread.start()
