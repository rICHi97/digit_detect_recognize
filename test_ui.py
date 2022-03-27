# -*- coding: utf-8 -*-
"""
Created on 2022-01-20 02:36:04

@author: Li Zhi
"""
import sys

from PyQt5 import QtWidgets, QtWebChannel
QWebChannel = QtWebChannel.QWebChannel
from pkgs.ui import ui_app, shared_core

QApplication = QtWidgets.QApplication

Inspection = ui_app.Inspection
ManageSystem = ui_app.ManageSystem
SharedCore = shared_core.SharedCore

# TODO：研究是否有别的方式显示网页
# TODO：第一次运行完成后，不能再次运行
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_system_manage = ManageSystem()
    # app_system_manage.main_window.update_db_display('Operator', ['张三', '1'])
    app_system_manage.show()
    sys.exit(app.exec_())
