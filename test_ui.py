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
SystemManage = ui_app.SystemManage
SharedCore = shared_core.SharedCore

# TODO：研究是否有别的方式显示网页
# TODO：第一次运行完成后，不能再次运行
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_inspection = Inspection()
    app_system_manage = SystemManage()
    app_inspection.connect['web_finish_username_pwd'](app_system_manage.update_label)
    app_system_manage.show()
    app_inspection.show()
    sys.exit(app.exec_())
