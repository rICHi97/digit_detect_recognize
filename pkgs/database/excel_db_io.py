# -*- coding: utf-8 -*-
"""
Created on 2021-11-28 04:08:30

@author: Li Zhi
读取xlx/xlsx表格，对每行数据加工后，调用my_database.store转为sqlite数据库
两类表格，变电站人员表，变电站端子表（优先）
"""
import os
from os import path

import pandas as pd

from . import cfg, data_factory, my_database

_excel_paths = cfg.excel_paths
_excel_types = cfg.excel_types
DataFactory = data_factory.DataFactory


# TODO：ui界面
# TODO：转为h5文件，比较差异行，更新数据
class Excel(object):
    """
    表格基础类
    """
    def __init__(self, excel_path, excel_type):
        with pd.ExcelFile(excel_path) as xlsx:
            # 默认表格只含1张sheet，第一行作为header，无index
            # 无需usecols参数，在后续加工中调用对应数据
            assert excel_type in _excel_types, f'excel_type不能为{excel_type}'
            self.df = pd.read_excel(xlsx).dropna(how='all')
            self.excel_type = excel_type

def _excel2db(excel):
    # TODO：pd的apply的func只接受df作为第一个参数？
    # excel -> normative_df -> model
    # normative_df规范统一表格数据格式
    normative_df = excel.df.apply(
        DataFactory.normalize,
        axis=1,
        excel_type=excel.excel_type,
    )
    model_data_dict = DataFactory.create_data(normative_df, excel.excel_type)
    my_database.store(model_data_dict)

def excel2db():
    """
    从表生成数据库
    每次执行会删除之前数据库，并读取所有表
    Parameters
    ----------
    Returns
    ----------
    """
    if path.exists(cfg.database_path):
        os.remove(cfg.database_path)
    my_database.create_tables()
    for excel_type in ('端子信息', ):
        excel = Excel(_excel_paths[excel_type], excel_type)
        _excel2db(excel)

@staticmethod
def db2excel(mdoel_type):
    pass

if __name__ == '__main__':
    pass
