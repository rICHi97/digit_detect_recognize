# -*- coding: utf-8 -*-
"""
Created on 2021-11-28 04:08:30

@author: Li Zhi
"""
from os import path

import pandas as pd


# TODO：读入txt数据创建表
# 该类围绕一个DataFrame及其excel文件
# TODO：事实上一个excel可以对应多个DF和sheet_name
class Excel(object):

    def __init__(self, excel_name, excel_dir='./', df=None, df_columns=None):
        self.excel_name = excel_name
        self.excel_dir = excel_dir
        self.excel_path = path.join(self.excel_dir, self.excel_name)
        self.df = df
        self.df_columns = df_columns
        if isinstance(self.df, pd.DataFrame):
            self.df.columns = self.df_columns

    def set_df(self, df: pd.DataFrame):
        self.df = df

    def set_columns(self, columns):
        self.df_columns = columns
        if isinstance(self.df, pd.DataFrame):
            self.df.columns = self.df_columns

    def to_excel(self, sheet_name):
        with pd.ExcelWriter(self.excel_path) as writer:  #pylint: disable=E0110
            self.df.to_excel(writer, sheet_name=sheet_name, index=False)

def to_excel():

    split_space_list = lambda line: [_ for _ in line.strip('').split(' ') if _ is not '']
    with open('test_data.txt', encoding='utf-8') as t:
        lines = t.readlines()
    columns = split_space_list(lines[1])
    df = []
    for row in lines[3:]:
        row = split_space_list(row)
        df.append(row)
    df = pd.DataFrame(df)
    xlsx = Excel('test_data.xlsx', df=df, df_columns=columns)
    xlsx.to_excel('sheet_1')

if __name__ == '__main__':
    to_excel()