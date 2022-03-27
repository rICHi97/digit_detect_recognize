# -*- coding: utf-8 -*-
"""
Created on 2022-03-22 10:14:21

@author: Li Zhi
"""
from PyQt5 import QtWebChannel

QWebChannelAbstractTransport = QtWebChannel.QWebChannelAbstractTransport


# transport的send_message方法调用
# client_wrapper的clientConnected信号连接WebChannel
# clientConnected
class WebSocketTransport(QWebChannelAbstractTransport):

    def __init__(self, my_socket):
        self.my_socket = my_socket
        # QWebChannelAbstractTransport.__init__(socket)
        self.my_socket.textMessageReceived.connect(self.text_message_received)

    def send_message(self, message):


    def text_message_received():
        pass