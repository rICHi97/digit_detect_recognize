# -*- coding: utf-8 -*-
"""
Created on 2021-12-25 22:13:42

@author: Li Zhi
重新实现某些qt类，自定义方法
"""
from PyQt5 import QtCore, QtWidgets

pyqtSignal = QtCore.pyqtSignal
QObject = QtCore.QObject
QDialog = QtWidgets.QDialog

# TODO：研究是否有更好的方法
# 接口类实现某个qt事件
# 需求：多个qt类可能需要重新实现相同的事件；一个qt类可能需要重新实现多个事件
# 单独重新实现close没有意义，不同的类可能需要不同的重新实现方法
# class ReimplementClose(QObject):


class UserDialog(QDialog):
    """
    重新实现close事件
    """
    # 必须作为类常量定义
    close_signal = pyqtSignal()

    def __init__(self, parent):
        # QDialog需要QWidget作为parent初始化，如果为None可能是调用了父类的初始化
        # QDialog.__init__(parent)
        super().__init__(parent)

    def closeEvent(self, event):  #pylint: disable=C0116
        self.close_signal.emit()
        QDialog.closeEvent(self, event)

