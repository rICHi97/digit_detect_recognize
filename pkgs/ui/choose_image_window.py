# -*- coding: utf-8 -*-
"""
Created on 2021-12-14 14:25:38

@author: Li Zhi
"""
# TODO：dialog可以封装成类
import sys
from os import path

from PyQt5 import QtWidgets

# 非运行入口时需要修改
from .qt_designer_code import Ui_choose_image_window
from . import reimplement

QApplication = QtWidgets.QApplication
QMessageBox = QtWidgets.QMessageBox
QFileDialog = QtWidgets.QFileDialog

Ui_Dialog = Ui_choose_image_window.Ui_Dialog
UserDialog = reimplement.UserDialog

class ChooseIMGWindow():
    """
    选择图片
    """
    def __init__(self, parent=None):
        self.img_type = None
        self.img_path = None
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
        self.connect = {
            'finish_choose_img': self.dialog.close_signal.connect,
        }
        self.ui_dialog.pushButton_3.clicked.connect(self.on_btn_clicked)
        self.ui_dialog.pushButton_4.clicked.connect(self.on_btn_clicked)

    def show(self):  #pylint: disable=C0116
        self.dialog.show()
        
    def close(self):  #pylint: disable=C0116
        self.dialog.close()

    def _get_img_type(self):
        if self.ui_dialog.radioButton.isChecked():
            self.img_type = '相机'
        elif self.ui_dialog.radioButton_2.isChecked():
            self.img_type = '相册'
        else:
            self.img_type = None

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
            self._get_img_type()
            assert self.img_type in('相机', '相册')
            # 打开文件对话框\
            # TODO：(*.jpg, *.png)只显示png图片，不显示jpg
            if self.img_type == '相册':
                img_path, _ = QFileDialog.getOpenFileName(
                    self.dialog, 
                    '选择要识别的图片',
                    './',
                    '图片(*.jpg *.png);;文本文件(*.txt)',
                )
            if img_path:
                self.img_path = img_path
                # 显示添加成功，直接关闭本对话框
                QMessageBox.information(
                    self.dialog,
                    '选择图片成功',
                    f'已成功添加{self.img_type}图片\n{path.basename(self.img_path)}',
                    QMessageBox.Ok,
                )
            else:
                QMessageBox.warning(
                    self.dialog,
                    '选择图片失败',
                    '未添加图片',
                    QMessageBox.Ok,
                )

        elif sender.text() == '退出':
            self.ui_dialog.radioButton.setChecked(False)
            self.ui_dialog.radioButton_2.setChecked(False)

        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    choose_img_window = ChooseIMGWindow()
    choose_img_window.show()
    sys.exit(app.exec_())