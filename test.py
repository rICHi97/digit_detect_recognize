 # -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:24:20 2021

@author: LIZHi
"""
import time

from pkgs.tool import code_testing

CodeTest = code_testing.CodeTest
# TODO：编号10有问题A
start = time.process_time()
CodeTest.test_recognize()
end = time.process_time()
print(end - start)
