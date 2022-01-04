# -*- coding: utf-8 -*-
"""
Created on 2021-12-25 01:54:18

@author: Li Zhi
"""
import sys

from PyQt5 import QtWidgets

from pkgs.ui import main_app

QApplication = QtWidgets.QApplication

MainApp = main_app.MainApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
    