# -*- coding: utf-8 -*-
"""
Created on 2021-12-13 21:21:09

@author: Li Zhi
"""
import sys
from os import path

from PyQt5 import QtWidgets

# 非运行入口时需要修改
from .qt_designer_code import Ui_connect_database_window
from . import reimplement

QApplication = QtWidgets.QApplication
QMessageBox = QtWidgets.QMessageBox
QFileDialog = QtWidgets.QFileDialog

Ui_Dialog = Ui_connect_database_window.Ui_Dialog
UserDialog = reimplement.UserDialog

class ConnectDBWindow():

    def __init__(self, parent=None):
        self.db_type = None
        self.db_path = None
        self._init_ui()
        self._init_content(parent)
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_dialog = Ui_Dialog()

    # 实体
    def _init_content(self, parent):
        self.dialog = UserDialog(parent)

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_dialog.setupUi(self.dialog)

    # 一个window的signal仅能处理本窗口中的事务
    # 如果需要涉及别的窗口，将其信号封装，在ui_integrated中连接
    def _setup_signal(self):
        # TODO：选择数据库完成时提供一个连接
        self.connect = {
            'connected_db': self.ui_dialog.pushButton.clicked.connect,
            'finish_connect_db': self.dialog.close_signal.connect,
        }
        self.ui_dialog.pushButton.clicked.connect(self.on_btn_clicked)
        self.ui_dialog.pushButton_2.clicked.connect(self.on_btn_clicked)

    def show(self):  #pylint: disable=C0116
        self.dialog.show()
        
    def close(self):  #pylint: disable=C0116
        self.dialog.close()

    def _get_db_type(self):
        if self.ui_dialog.radioButton.isChecked():
            self.db_type = '本地'
        elif self.ui_dialog.radioButton_2.isChecked():
            self.db_type = '网络'
        else:
            self.db_type = None

    def on_btn_clicked(self):
        """
        处理确认按键和退出按键
        若确认，根据数据库类型选择数据库连接；
        若退出，退出本窗口
        Parameters
        ----------
        Returns
        ----------
        """
        sender = self.dialog.sender()
        assert sender.text() in ('确认', '退出')
        if sender.text() == '确认':
            # TODO：未选择db_type时
            self._get_db_type()
            assert self.db_type in('网络', '本地'), f'db_type错误，不能为{db_type}'
            # 打开文件对话框
            if self.db_type == '本地':
                db_path, _ = QFileDialog.getOpenFileName(
                    self.dialog,
                    '选择要连接的数据库',
                    './',
                    '数据库(*.db);;文本文件(*.txt);;表格(*.xlsx, *.xls)',
                )
            if db_path:
                self.db_path = db_path
                # 显示添加成功，直接关闭本对话框
                QMessageBox.information(
                    self.dialog,
                    '连接成功',
                    f'已成功连接到{self.db_type}数据库\n{path.basename(self.db_path)}',
                    QMessageBox.Ok,
                )
            else:
                QMessageBox.warning(
                    self.dialog,
                    '连接失败',
                    '未选择数据库',
                    QMessageBox.Ok,
                )
        elif sender.text() == '退出':
            self.ui_dialog.radioButton.setChecked(False)
            self.ui_dialog.radioButton_2.setChecked(False)

        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    connect_db_window = ConnectDBWindow()
    connect_db_window.show()
    sys.exit(app.exec_())