# -*- coding: utf-8 -*-
"""
Created on 2022-01-20 00:42:32

@author: Li Zhi
"""
from PyQt5 import QtCore

QThread = QtCore.QThread
QMutex = QtCore.QMutex

# TODO：提供中止功能
class DetectRecognizeThread(QThread):
    """
    该线程负责完成端到端的文本检测及识别
    """
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.end_to_end = None
        self.graph = None
        self.img_path = None

    # TODO：是否可以仅提供detect方法的引用
    def my_start(self, end_to_end, graph, img_path):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        self.mutex.lock()
        self.end_to_end = end_to_end
        self.graph = graph
        # self.graph = end_to_end.get_graph()
        self.img_path = img_path
        self.mutex.unlock()
        self.start()

    def run(self):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        self.mutex.lock()
        end_to_end = self.end_to_end
        graph = self.graph
        img_path = self.img_path
        self.mutex.unlock()
        with graph.as_default():
            end_to_end.detect_recognize(img_path)


# TODO：后续可能需要在服务器上实现
class LoadThread(QThread):
    """
    用于初始化加载
    """
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.end_to_end = None
        self.graph = None

    def run(self):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        from .. import detect_recognize  #pylint: disable=C0415
        self.mutex.lock()
        end_to_end = detect_recognize.EndToEnd()
        graph = end_to_end.get_graph()
        self.end_to_end = end_to_end
        self.graph = graph
        self.mutex.unlock()
