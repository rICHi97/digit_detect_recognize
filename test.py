 # -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:24:20 2021

@author: LIZHi
"""
import time

from pkgs.tool import code_testing

start = time.process_time()

CodeTest = code_testing.CodeTest
CodeTest.test_joint_rec()

end = time.process_time()
print(end - start)
