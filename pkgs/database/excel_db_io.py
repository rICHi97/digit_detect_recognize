# -*- coding: utf-8 -*-
"""
Created on 2021-11-28 04:08:30

@author: Li Zhi
读取xlx/xlsx表格，对每行数据加工后，调用my_database.store转为sqlite数据库
两类表格，变电站人员表，变电站端子表（优先）
"""
from os import path

import pandas as pd

from . import cfg, data_factory, my_database

df_row_to_dict = data_factory.df_row_to_dict
excel_paths = cfg.excel_paths

# TODO：ui界面
# TODO：转为h5文件，比较差异行，更新数据
class Excel(object):
    """
    基类，提供
    """
    def __init__(self, excel_path, model_type):
        with pd.ExcelFile(excel_path) as xlsx:
            # 默认表格只含1张sheet，第一行作为header，无index
            # 无需usecols参数，在后续加工中调用对应数据
            self.df = pd.read_excel(xlsx)
            self.model_type = model_type

def _excel2db(excel, model_type):
    if model_type == 'Terminal':
        # TODO：pd的apply的func只接受df作为第一个参数？
        series = excel.df.apply(
            df_row_to_dict,
            axis=1,
            model_type=model_type,
        )
        for _ in series:
            print(_['terminal_id'])
            print(_['loop_number'])
            my_database.store(
                model_type,
                terminal_id=_['terminal_id'],
                loop_number=_['loop_number'],
            )
        # TODO：迭代操作是否需要优化 

def excel2db():
    """
    从表生成数据库
    每次执行会删除之前数据库，并读取所有表
    Parameters
    ----------
    Returns
    ----------
    """
    # TODO：清空指定位置的db文件
    my_database.create_tables()
    for model_type in ('Terminal', ):
        excel = Excel(excel_paths[model_type], model_type)
        _excel2db(excel, model_type)
    my_database.close_db()

@staticmethod
def db2excel(mdoel_type):
    pass

if __name__ == '__main__':
    pass
