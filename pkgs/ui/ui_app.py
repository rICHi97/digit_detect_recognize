# -*- coding: utf-8 -*-
"""
Created on 2021-12-13 15:28:08

@author: Li Zhi
集成多个app类
"""
from os import path

from PyQt5 import QtCore, QtNetwork, QtWidgets, QtWebChannel, QtWebEngineWidgets, QtWebSockets

from . import cfg, my_thread, shared_core
from .window import manage_system
from .. database import data_retrieving, data_factory, excel_db_io

QFileInfo = QtCore.QFileInfo
QUrl = QtCore.QUrl
QHostAddress = QtNetwork.QHostAddress
QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QWebEngineView = QtWebEngineWidgets.QWebEngineView
QWebChannel = QtWebChannel.QWebChannel
QWebSocketServer = QtWebSockets.QWebSocketServer

DetectRecognizeThread = my_thread.DetectRecognizeThread
SharedCore = shared_core.SharedCore
LoadThread = my_thread.LoadThread
DataRetrieve = data_retrieving.DataRetrieve
DataFactory = data_factory.DataFactory

RESULT_IMG_PATH = './resource/tmp.jpg'


# TODO：退出程序时确保关闭数据库
class ManageSystem():

    MainWindow = manage_system.MainWindow
    ReleaseTaskWindow = manage_system.ReleaseTaskWindow

    def __init__(self):
        self._init_ui()
        self._setup_signal()
        self.server = QWebSocketServer('Manage System Server', QWebSocketServer.NonSecureMode)
        if not self.server.listen(QHostAddress.LocalHost, 12345):
            print('打开web socket服务器失败')

    def _init_ui(self):
        self.main_window = self.MainWindow()
        self.release_task_window = self.ReleaseTaskWindow(self.main_window.main_window)

    def _setup_signal(self):
        self.main_window.connect['create_database'](self.create_database)
        self.main_window.connect['connect_database'](self.connect_database)
        self.main_window.connect['release_task'](self.release_task_window.show)

    def create_database(self):
        excel_db_io.excel2db()
        # TODO：消息框
        print('创建数据库成功')

    def connect_database(self):
        self.main_window.update_db_status('connected', '已连接到数据库')
        all_operatos = DataRetrieve.all_operators()
        self.update_db_display('Operator', all_operatos)
        all_components = DataRetrieve.all_components()
        combobox_data = {'Operator': all_operatos, 'Component': all_components}
        self.release_task_window.update_combobox(combobox_data)
        print('连接数据库成功')


    def update_db_display(self, table_type, datas):
        for data in datas:
            self.main_window.update_db_display(table_type, data)

    def show(self):  #pylint: disable=C0116
        self.main_window.show()


class Inspection():

    def __init__(self, url=cfg.inspection_web):
        # super(Inspection, self).__init__()
        self.url = url
        self._init_ui()
        self._setup_signal()

    def _init_ui(self):
        self.view = QWebEngineView()
        self.shared_core = SharedCore()
        self.channel = QWebChannel()
        self.channel.registerObject('sharing', self.shared_core)
        self.view.page().setWebChannel(self.channel)
        self.view.load(QUrl(QFileInfo(self.url).absoluteFilePath()))

    def _setup_signal(self):
        self.connect = {
            'web_finish_username_pwd': self.shared_core.finish[list].connect,
        }

    def show(self):
        self.view.show()

    def __del__(self):
        self.view.deleteLater()

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # main_app = MainApp()
    # main_app.show()
    # sys.exit(app.exec_())
    print('ui_app.py')
