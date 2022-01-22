# -*- coding: utf-8 -*-
"""
Created on 2022-01-16 16:35:31

@author: Li Zhi
"""
import socket

from .. import detect_recognize

EndToEnd = detect_recognize.EndToEnd


# TODO：cfg参数化
# 创建一个服务，接受输入图片路径，返回
class Server():

    def __init__(self):
        HOST = 'localhost'
        PORT = 50007
        self.end_to_end = EndToEnd()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.socket.listen(1)
        while 1:
            conn, addr = self.socket.accept()
            print(f'Connected by {addr}')
            bytes_img_path = conn.recv(1024)
            if not bytes_img_path:
                break
            img_path = str(bytes_img_path, encoding='utf-8')
            self.end_to_end.test_predict(img_path)
            conn.sendall(b'test')
            conn.close()
