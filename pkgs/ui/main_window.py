# -*- coding: utf-8 -*-
"""
Created on 2021-12-12 04:40:40

@author: Li Zhi
"""
# -*- coding: utf-8 -*-
"""
Ui_window_name.py为.ui文件编译，不应手动改动
window_name.py为基于Ui的模块
ui_window_name为ui实例，绑定了各种widget
window_name为QMainWindow对象，通过setupUi装载到ui_window_name中
"""
import sys

from PyQt5 import QtWidgets, QtGui

from .qt_designer_code import Ui_main_window

QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QPixmap = QtGui.QPixmap

Ui_MainWindow = Ui_main_window.Ui_MainWindow


# TODO：结构优化
class MainWindow():

    # TODO：执行setup和show之前检查变量
    def __init__(self):
        self._init_ui()
        self._init_content()
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_main_window = Ui_MainWindow()

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
        self.connect = {
            'clicked_connect_db': self.ui_main_window.pushButton.clicked.connect,
            'clicked_choose_img': self.ui_main_window.pushButton_2.clicked.connect,
            'clicked_start_recognize': self.ui_main_window.pushButton_3.clicked.connect,
        }

    def show(self):  #pylint: disable=C0116
        self.main_window.show()
        self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_disconnected"))

    def close(self):  #pylint: disable=C0116
        self.main_window.close()

    def on_btn_clicked(self):
        pass

    def set_terminal_img(self, img_path):
        """
        设置显示区域所显示的端子排图片
        Parameters
        ----------
        Returns
        ----------
        """
        # TODO：设置铺满
        self.ui_main_window.label_4.setPixmap(QPixmap(img_path))
        self.ui_main_window.label_4.setScaledContents(True)

    def set_db_status(self, status, text):
        """
        设置程序下方数据库状态信息
        Parameters
        ----------
        Returns
        ----------
        """
        assert status in ('connected', 'disconnected')
        if status == 'connected':
            self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_connected"))
        else:
            self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_disconnected"))
        self.ui_main_window.label.setText(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())
