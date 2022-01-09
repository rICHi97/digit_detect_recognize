 # -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:24:20 2021

@author: LIZHi
"""
import time

from pkgs.tool import code_testing

CodeTest = code_testing.CodeTest

start = time.process_time()
# 检查label生成过程与nms
# CodeTest.test_preprocess()
# CodeTest.test_label()
CodeTest.test_draw_gt()

end = time.process_time()
print(end - start)
    

