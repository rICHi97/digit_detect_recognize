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

excel_paths = cfg.excel_paths
excel_types = cfg.excel_types
excel_args = cfg.excel_args

DataFactory = data_factory.DataFactory


# TODO：ui界面
# TODO：转为h5文件，比较差异行，更新数据
class Excel(object):
    """
    表格基础类，返回包含多张标准格式df的dict
    """
    def __init__(self, excel_path, excel_type):

        assert excel_type in excel_types, f'excel_type不能为{excel_type}'

        sheet_args = excel_args[excel_type]

        # {'Cubicle': '计量柜信息表', }
        sheet_names = {}
        for key, value in sheet_args.items():
            sheet_names[key] = value['sheet_name']

        # 默认表格第一行作为header，无index
        with pd.ExcelFile(excel_path) as xlsx:
            df_dict = pd.read_excel(xlsx, sheet_name=list(sheet_names.values()))

        self.df_dict = {}
        for key, value in sheet_args.items():

            # 形参和实参，按实参冲df取列，转为以形参为标准列名的df
            columns_param = [column for column in value.keys() if column != 'sheet_name']
            columns_arg = [value[param] for param in columns_param]

            sheet_name = value['sheet_name']
            df = df_dict[sheet_name][columns_arg].dropna(how='all')
            df.columns = columns_param
            self.df_dict[key] = pd.DataFrame(df, columns=columns_param)

        self.excel_type = excel_type

def _excel2db(excel):
    model_data_dict = DataFactory.create_data(excel)
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
    # 移除之前数据库文件
    if path.exists(cfg.database_path):
        my_database.close()
        os.remove(cfg.database_path)
    my_database.create_tables()
    for excel_type in ('二次回路信息表', ):
        excel = Excel(excel_paths[excel_type], excel_type)
        _excel2db(excel)

@staticmethod
def db2excel(mdoel_type):
    pass

if __name__ == '__main__':
    pass
