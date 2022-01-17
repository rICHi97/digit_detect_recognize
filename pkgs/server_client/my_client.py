# -*- coding: utf-8 -*-
"""
Created on 2022-01-16 16:51:31

@author: Li Zhi
"""
import socket

HOST = 'localhost'    # The remote host
PORT = 50007              # The same port as used by the server


class Client():

    def __init__(self):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test(self, img_path):
        self.socket.connect((HOST, PORT))
        self.socket.sendall(img_path)
        data = self.socket.recv(1024)
        self.socket.close()
        print(f'Received{repr(data)}')
