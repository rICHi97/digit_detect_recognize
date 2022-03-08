# -*- coding: utf-8 -*-
"""
Created on 2022-02-28 17:59:52

@author: Li Zhi
"""
import sys
from os import path

from PyQt5 import QtCore, QtGui, QtWidgets

from ..qt_designer_code import Ui_choose_image_window, Ui_connect_database_window, Ui_init_window, Ui_main_window  #pylint: disable=C0301
from .. import my_thread, reimplement

QTimer = QtCore.QTimer
QPixmap = QtGui.QPixmap
QApplication = QtWidgets.QApplication
QDialog = QtWidgets.QDialog
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox

LoadThread = my_thread.LoadThread
UserDialog = reimplement.UserDialog


class ChooseIMGWindow():
    """
    选择图片
    """
    Ui_Dialog = Ui_choose_image_window.Ui_Dialog

    def __init__(self, parent=None):
        self.img_type = None
        self.img_path = None
        self._init_ui()
        self._init_content(parent)
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_dialog = self.Ui_Dialog()

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


class ConnectDBWindow():

    Ui_Dialog = Ui_connect_database_window.Ui_Dialog

    def __init__(self, parent=None):
        self.db_type = None
        self.db_path = None
        self._init_ui()
        self._init_content(parent)
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_dialog = self.Ui_Dialog()

    # 实体
    def _init_content(self, parent):
        self.dialog = UserDialog(parent)

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_dialog.setupUi(self.dialog)

    # 一个window的signal仅能处理本窗口中的事务
    # 如果需要涉及别的窗口，将其信号封装，在ui_integrated中连接
    def _setup_signal(self):
        # TODO：选择数据库完成时提供一个连接
        self.connect = {
            'connected_db': self.ui_dialog.pushButton.clicked.connect,
            'finish_connect_db': self.dialog.close_signal.connect,
        }
        self.ui_dialog.pushButton.clicked.connect(self.on_btn_clicked)
        self.ui_dialog.pushButton_2.clicked.connect(self.on_btn_clicked)

    def show(self):  #pylint: disable=C0116
        self.dialog.show()

    def close(self):  #pylint: disable=C0116
        self.dialog.close()

    def _get_db_type(self):
        if self.ui_dialog.radioButton.isChecked():
            self.db_type = '本地'
        elif self.ui_dialog.radioButton_2.isChecked():
            self.db_type = '网络'
        else:
            self.db_type = None

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
            # TODO：未选择db_type时
            self._get_db_type()
            assert self.db_type in('网络', '本地'), f'db_type错误，不能为{db_type}'
            # 打开文件对话框
            if self.db_type == '本地':
                db_path, _ = QFileDialog.getOpenFileName(
                    self.dialog,
                    '选择要连接的数据库',
                    './',
                    '数据库(*.db);;文本文件(*.txt);;表格(*.xlsx, *.xls)',
                )
            if db_path:
                self.db_path = db_path
                # 显示添加成功，直接关闭本对话框
                QMessageBox.information(
                    self.dialog,
                    '连接成功',
                    f'已成功连接到{self.db_type}数据库\n{path.basename(self.db_path)}',
                    QMessageBox.Ok,
                )
            else:
                QMessageBox.warning(
                    self.dialog,
                    '连接失败',
                    '未选择数据库',
                    QMessageBox.Ok,
                )
        elif sender.text() == '退出':
            self.ui_dialog.radioButton.setChecked(False)
            self.ui_dialog.radioButton_2.setChecked(False)

        self.close()


# TODO：位置变动后，主窗口跟随
class InitWindow():
    """
    初始化
    """
    Ui_Dialog = Ui_init_window.Ui_Dialog

    def __init__(self):
        self._init_ui()
        self._init_content()
        self._setup_ui()
        self._setup_signal()

    # ui框架
    def _init_ui(self):
        self.ui_dialog = self.Ui_Dialog()

    # 实体
    def _init_content(self):
        self.dialog = QDialog()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.load_thread = LoadThread()
        self.end_to_end = None
        self.graph = None

    # 给ui框架填充实体
    def _setup_ui(self):
        self.ui_dialog.setupUi(self.dialog)

    def _setup_signal(self):
        self.load_thread.finished.connect(self.update_status)
        self.timer.timeout.connect(self.dialog.close)
        self.connect = {
            'init_finished': self.timer.timeout.connect,
        }


    # TODO：除更新文本外也更新图片
    def update_status(self):
        """
        更新文本状态
        Parameters
        ----------
        Returns
        ----------
        """
        self.end_to_end = self.load_thread.end_to_end  #pylint: disable=W0201
        self.graph = self.load_thread.graph  #pylint: disable=W0201
        self.ui_dialog.label_3.setText(
            "<html><head/><body><p><font size=\"5\", face=\"黑体\">"
            "初始化完成.</font></p></body></html>"
        )
        self.timer.start(1500)

    def show(self):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        self.dialog.show()
        self.load_thread.start()


# TODO：结构优化
class MainWindow():

    Ui_MainWindow = Ui_main_window.Ui_MainWindow

    # TODO：执行setup和show之前检查变量
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

    # 一个window的signal仅能处理本窗口中的事务
    # 如果需要涉及别的窗口，将其信号封装，在ui_integrated中连接
    # MainWindow().connect['clicked_connect_db'](callable_object)
    def _setup_signal(self):
        self.connect = {
            'clicked_connect_db': self.ui_main_window.pushButton.clicked.connect,
            'clicked_choose_img': self.ui_main_window.pushButton_2.clicked.connect,
            'clicked_start_recognize': self.ui_main_window.pushButton_3.clicked.connect,
        }

    def show(self):  #pylint: disable=C0116
        self.main_window.show()
        # TODO：show了之后才能setPixmap？
        self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_disconnected"))

    def set_terminal_img(self, img_path):
        """
        设置显示区域所显示的端子排图片
        Parameters
        ----------
        Returns
        ----------
        """
        self.ui_main_window.label_4.setPixmap(QPixmap(img_path))
        self.ui_main_window.label_4.setScaledContents(True)

    def set_db_status(self, status, text):
        """
        设置程序下方数据库状态信息
        Parameters
        ----------
        Returns
        ----------
        """
        assert status in ('connected', 'disconnected')
        if status == 'connected':
            self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_connected"))
        else:
            self.ui_main_window.label_2.setPixmap(QPixmap(":/mainwindow/img/db_disconnected"))
        self.ui_main_window.label.setText(text)

if __name__ == '__main__':
    print('pkgs.ui.window.detect_recognize')