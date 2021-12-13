# -*- coding: utf-8 -*-
"""
Created on 2021-12-13 15:28:08

@author: Li Zhi
"""
import sys

from PyQt5 import QtWidgets

import connect_database_window
import main_window

QApplication = QtWidgets.QApplication

ConnectDBWindow = connect_database_window.ConnectDBWindow
MainWindow = main_window.MainWindow

class MainApp():

    def __init__(self):

        self.connect_db_window = ConnectDBWindow()
        self.main_window = MainWindow()
        self._setup_signal()

    def _setup_signal(self):
        self.main_window.connect['clicked_connect_db'](
            self.connect_db_window.show
        )

    def show(self):  #pylint: disable=C0116
        self.main_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
