# -*- coding: utf-8 -*-
"""
Created on 2022-01-16 16:57:22

@author: Li Zhi
"""
import time

from pkgs.server_client import my_client

Client  = my_client.Client
# Server = my_server.Server

time1 = time.process_time()
client = Client()
time2 = time.process_time()
client.test(b'1.jpg')
time3 = time.process_time()
A