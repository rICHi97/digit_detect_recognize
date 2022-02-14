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

normalize_excel = data_factory.normalize_excel
excel_paths = cfg.excel_paths
# 区别于model_type，需要从一张excel中拆分出多个model
_excel_types = cfg.excel_types


# TODO：ui界面
# TODO：转为h5文件，比较差异行，更新数据
class Excel(object):
    """
    基类，提供
    """
    # TODO：df去除全空行
    def __init__(self, excel_path, excel_type):
        with pd.ExcelFile(excel_path) as xlsx:
            # 默认表格只含1张sheet，第一行作为header，无index
            # 无需usecols参数，在后续加工中调用对应数据
            self.df = pd.read_excel(xlsx)
            self.excel_type = excel_type

def _excel2db(excel, excel_type):
    assert excel_type in _excel_types, f'excel_type不能为{excel_type}'
    # TODO：pd的apply的func只接受df作为第一个参数？
    # excel -> normative_df -> model
    # normative_df规范统一表格数据格式
    normative_df = excel.df.apply(
        normalize_excel,
        axis=1,
        excel_type=excel_type,
    )

    if excel_type == '端子信息':
        cubicles = normative_df['cubicle_id'].unique()
        # dict_ = {
        #     cubicle: {
        #         install_unit: {
        #             terminal: {
        #                 'loop'
        #             }
        #         }

        #     }
        # }
        pass

    return normative_df

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
    for excel_type in ('端子信息', ):
        excel = Excel(excel_paths[excel_type], excel_type)
        _excel2db(excel, excel_type)
    my_database.close_db()

@staticmethod
def db2excel(mdoel_type):
    pass

if __name__ == '__main__':
    pass
