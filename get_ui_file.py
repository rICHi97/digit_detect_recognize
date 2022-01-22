# -*- coding: utf-8 -*-
"""
Created on 2022-01-20 02:19:56

@author: Li Zhi
"""
from pkgs.tool import tool

UiTool = tool.UiTool


if __name__ == '__main__':
    UiTool.ui_qrc_to_py()
    UiTool.update_qt_designer_code()
