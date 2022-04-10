# -*- coding: utf-8 -*-
"""
Created on 2021-12-13 18:25:46

@author: Li Zhi
本模块主要提供对文件的操作。
"""
import os
from os import path
import subprocess
import shutil

from . import cfg


class UiTool():
    """
    """
    @staticmethod
    def ui_qrc_to_py(
        qrc_path=cfg.qrc_path, output_dir=cfg.qrc_output_dir, delete_qrc=cfg.delete_qrc,
    ):
        """
        调用命令行将资源文件转为py
        资源文件.qrc仅包括对所需资源的描述
        转为py需要qrc能够访问所需资源
        转为py后可仅保存py文件
        Parameters
        ----------

        Returns
        ----------
        """
        qrc_filename = path.basename(qrc_path)[:-4]
        py_name = qrc_filename + '_rc.py'
        py_path = path.join(output_dir, py_name)
        expr = f'pyrcc5 -o {py_path} {qrc_path}'
        subprocess.run(expr)  #pylint: disable=W1510
        # 当在ui文件夹内运行时需要
        # expr = f'pyrcc5 -o ./ui/ {qrc_path}'
        # subprocess.run(expr)
        if delete_qrc:
            os.remove(qrc_path)

    @staticmethod
    def update_qt_designer_code(
        old_code_dir=cfg.old_code_dir,
        new_code_dir=cfg.new_code_dir,
        keyword=cfg.update_keyword,
    ):
        """
        使用qt workspace中的ui.py更新本项目中的ui.py
        Parameters
        ----------

        Returns
        ----------
        """
        new_files = os.listdir(new_code_dir)
        new_files = [file for file in new_files if keyword in file and '.py' in file]
        for file in new_files:
            new_file = path.join(new_code_dir, file)
            shutil.copy(new_file, old_code_dir)
