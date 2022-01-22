# -*- coding: utf-8 -*-
"""
Created on 2022-01-20 02:36:04

@author: Li Zhi
"""
import sys

from PyQt5 import QtWidgets

from pkgs.ui import init_window

QApplication = QtWidgets.QApplication
InitWindow = init_window.InitWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    initwindow = InitWindow()
    initwindow.show()
    sys.exit(app.exec_())
