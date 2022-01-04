# -*- coding: utf-8 -*-
"""
Created on 2021-12-29 01:34:54

@author: Li Zhi
"""
from pkgs.east import east_data, east_net

EastPreprocess = east_data.EastPreprocess
EastNet = east_net.EastNet

# TODO：检查参数，检查文件，训练一个能分辨类别的模型用于后续测试验证
# TODO：优化模型
# TODO：检查np限制计算精度能否加快速度
# TODO：先从mistgpu下载npy文件检查是否正确生成label
# python east_preprocess_train.py; sleep 300; sh /mistgpu/shutdown.sh
if __name__ == '__main__':
    # EastPreprocess.preprocess()
    # EastPreprocess.label()
    east = EastNet()
    # east.train()
    east.predict()
    