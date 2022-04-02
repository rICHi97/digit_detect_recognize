# -*- coding: utf-8 -*-
"""
Created on 2022-02-28 18:08:07

@author: Li Zhi
"""
from PyQt5 import QtGui, QtWidgets

from ..qt_designer_code import Ui_manage_system_main_window, Ui_manage_system_release_task_window
from ...database import data_factory

QPixmap = QtGui.QPixmap
QMainWindow = QtWidgets.QMainWindow
QTableWidgetItem = QtWidgets.QTableWidgetItem

DataFactory = data_factory.DataFactory


# TODO：禁止编辑表格区域
class MainWindow(object):
    """
    系统管理主窗口
    """
    Ui_MainWindow = Ui_manage_system_main_window.Ui_MainWindow

    def __init__(self):
        self._init_ui()
        self._init_content()
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_main_window = self.Ui_MainWindow()

    # 实体
    def _init_content(self):
        self.main_window = QMainWindow()

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_main_window.setupUi(self.main_window)
        self.tables = {
            'Operator': self.ui_main_window.tableWidget,
            'Cubicle': self.ui_main_window.tableWidget_2,
            'Component': self.ui_main_window.tableWidget_3,
            'Record': self.ui_main_window.tableWidget_4,
        }
        self.data_cnts = {
            'Operator': 0,
            'Cubicle': 0,
            'Component': 0,
            'Record': 0,
        }

    # 一个window的signal仅能处理本窗口中的事务
    # 如果需要涉及别的窗口，将其信号封装，在ui_integrated中连接
    # MainWindow().connect['clicked_connect_db'](callable_object)
    def _setup_signal(self):
        self.connect = {
            'create_database': self.ui_main_window.pushButton.clicked.connect, # 创建数据库
            'connect_database': self.ui_main_window.pushButton_2.clicked.connect, # 连接数据库
            'release_task': self.ui_main_window.pushButton_3.clicked.connect, # 创建检验工单
        }

    def show(self):
        self.main_window.show()

    def update_db_status(self, status, text):
        assert status in ('connected', 'disconnected')
        if status == 'connected':
            self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_connected"))
        else:
            self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_disconnected"))
        self.ui_main_window.label.setText(text)

    def update_db_display(self, table_type, data):
        """
        Parameters
        ----------
        table_type：one of ('Operator', 'Cubicle', 'Component', 'Record')
        data：list或元组，每列是要存入的数据

        Returns
        ----------
        """
        assert table_type in ('Operator', 'Cubicle', 'Component', 'Record')
        table = self.tables[table_type]
        data_cnt = self.data_cnts[table_type]
        # 表格显示行数等于已存数据条数，增加一行用于显示
        if data_cnt == table.rowCount():
            table.setRowCount(data_cnt + 1)
        # 数据行列基于0起始
        for i, column in enumerate(data):
            new_item = QTableWidgetItem(column)
            table.setItem(data_cnt, i, new_item)
        data_cnt += 1
        self.data_cnts[table_type] = data_cnt


    def update_task_status(self, text):
        self.ui_main_window.label_5.setText()


class ReleaseTaskWindow(object):
    """
    发布任务子窗口
    """
    Ui_MainWindow = Ui_manage_system_release_task_window.Ui_MainWindow

    def __init__(self, parent=None):
        self._init_ui()
        self._init_content(parent)
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_main_window = self.Ui_MainWindow()

    # 实体
    def _init_content(self, parent):
        self.main_window = QMainWindow(parent)

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_main_window.setupUi(self.main_window)


    def _setup_signal(self):
        self.connect = {
            'confirm_release': self.ui_main_window.pushButton.clicked.connect, # 确认发布工单
        }

    def update_combobox(self, combobox_data):
        operators = combobox_data['Operator']
        operators = [operator[0] for operator in operators]
        components = combobox_data['Component']
        three_phase_components = DataFactory.three_phase_components(components)
        TVs = three_phase_components['TV']
        PJs = three_phase_components['PJ']
        self.ui_main_window.comboBox.addItems(operators)
        self.ui_main_window.comboBox_2.addItems(TVs)
        self.ui_main_window.comboBox_3.addItems(PJs)

    def show(self):
        self.main_window.show()
