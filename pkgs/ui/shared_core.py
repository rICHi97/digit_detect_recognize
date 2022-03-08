# -*- coding: utf-8 -*-
"""
Created on 2022-02-26 15:21:12

@author: Li Zhi
该类的实例用于html与QWebChannel间的数据共享
"""
from PyQt5 import QtCore, QtWidgets

pyqtProperty = QtCore.pyqtProperty
pyqtSignal = QtCore.pyqtSignal
QWidget = QtWidgets.QWidget


class SharedCore(QWidget):
    """
    继承自QObject，实例用于html与QWebCh间共享传递数据
    传递数据包括用户名、密码；计量柜id；端子扫描图片；
    """

    # 信号
    # list参数类型用于发射username_pwd
    # str参数类型用于发射cubicle_id
    finish = pyqtSignal([list], [str], name='finish')

    def __init__(self):
        super().__init__()

    def get_username_pwd(self):
        return 'this is username_pwd'

    def web2qt_username_pwd(self, username_pwd):
        """
        当web端设置共享类的pyqtProperty时调用此方法，从而发射信号给qt端
        Parameters
        ----------
        username_pwd：用户名拼接密码，空格区分

        Returns
        ----------
        """
        username, pwd = username_pwd.split()[0], username_pwd.split()[1]
        self.finish[list].emit([username, pwd])

    def web2qt_cubicle_id(self, cubicle_id):
        """
        Parameters
        ----------
        cubicle_id：计量柜id

        Returns
        ----------
        """
        self.finish[str].emit(cubicle_id)

    # 当在web中设置值时，SharedCore.username_pwd = xx时将会调用web2qt_username_pwd方法
    username_pwd = pyqtProperty(str, fget=get_username_pwd, fset=web2qt_username_pwd)
    # cubicle_id = pyqtProperty(str, fset=web2qt_cubicle_id)
