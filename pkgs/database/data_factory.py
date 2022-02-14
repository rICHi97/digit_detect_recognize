# -*- coding: utf-8 -*-
"""
Created on 2022-01-30 15:52:53

@author: Li Zhi
在recdata模块，excel_db模块之间加工数据以符合要求
"""
import pandas as pd

from . import cfg

excel_types = cfg.excel_types


class DataFactory():

    # TODO：整型id是否能加快速度？
    @staticmethod
    def normalize_cubicle_id(excel_cubicle_id):
        """
        Parameters
        ----------
        excel_cubicle_id：
            计量柜号，2位PJ固定，1-2位种类，2-4位电压，1位结构类别，3位方案编号（1位英文，2位数字）
            种类：整体式1，分体式仪表2，分体式互感器2H
            电压：
                对于1，35/6~10/0.38kV。35kV，35；6kV，06（猜测）；0.38kV，038；
                对于2，固定为0.1kV，010；
                对于2H，6~35kV。

        Returns
        ----------
        normalized_cubicle_id：统一规范格式后计量柜id
        """
        e = str(excel_cubicle_id)
        # 操作计量柜号，前两位固定为'PJ'
        truncate_index = 3
        if '1' in e[2:4]:
            type_ = '01' # 整体式
        elif '2H' in e[2:4]:
            type_ = '2H' # 分体互感器
            truncate_index = 4
        else:
            type_ = '02' # 分体仪表
        voltage_class = e[truncate_index:-4]
        print(voltage_class)
        assert len(voltage_class) in (2, 3), '计量柜编号电压等级有误' # 2位或3位，35/06/038/010
        if len(voltage_class) == 2:
            # 转为10V单位，06(kV) -> 600(10V)，35(kV) -> 3500(10V)
            voltage_class = (
                100 * int(voltage_class[1]) if voltage_class[0] == '0' else 100 * int(voltage_class)
            )
        else:
            voltage_class = int(voltage_class[1:])
        # 转为10V，右对齐，左侧补零，宽4位
        voltage_class = f'{voltage_class:0>4}'
        structure, design = e[-4], e[-3:] # 1位结构类别，3位方案编号
        # 计量柜号，宽10位
        normative_cubicle_id = f'{type_}{voltage_class}{structure}{design}'

        return normative_cubicle_id

#    @staticmethod
#     def terminal_info_to_id(cibucle_num, install_unit_num, terminal_num):
#         """
#         Parameters
#         ----------
#         cibucle_num：
#         install_unit_num：安装单位号，1-2位
#         terminal_num：端子编号，1-3位

#         Returns
#         ----------
#         terminal_id：
#         """
#         c, i, t = str(cibucle_num), str(install_unit_num), str(terminal_num)
#         # 安装单位号，右对齐，左侧补0，宽2位
#         i_num = f'{i:0>2}'
#         # 端子编号，右对齐，左侧补0，宽3位
#         t_num = f'{t:0>3}'
#         terminal_id = f'{c_num}{i_num}{t_num}'

#         return terminal_id


# TODO：row键名cfg参数化
def normalize_excel(row, excel_type):
    """
    df_row -> dict，调用字典存入db
    Parameters
    ----------
    Returns
    ----------
    s：Series
        端子信息：pd.Series([cubicle_id, plate_text, terminal_num, loop_num])
    """
    assert excel_type in excel_types, f'excel_type不能为{excel_type}'
    column_names = cfg.excel_column_names
    if excel_type == '端子信息':
        column_name = column_names['cubicle_id']
        cubicle_id = row[column_name]
        n_cubicle_id = DataFactory.normalize_cubicle_id(cubicle_id)
        d = {'cubicle_id': n_cubicle_id}
        keys = ['plate_text', 'terminal_num', 'loop_num']

    for key in keys:
        column_name = column_names[key]
        d[key] = row[column_name]

    s = pd.Series(d)

    return s

