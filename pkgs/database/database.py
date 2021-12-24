# -*- coding: utf-8 -*-
"""
Created on 2021-11-28 02:37:33

@author: Li Zhi
本模块用于实现数据库的底层操作，创建表等
"""
from os import path
import sqlite3

import pandas as pd

import excel


class Database(object):
    """
    用于创建、维持数据库，并提供命令执行。
    """

    def __init__(self, database_name, database_dir='./'):
        """
        Create a Database object for interacting with the sqlite3 db file `database_name`. The Datbase object is automatically connected and ready to use.
        """
        self.database_name = database_name
        self.database_dir = database_dir
        self.database_path = path.join(self.database_dir, self.database_name)
        self.con = self.connect()

    def connect(self):
        """Return a connection to the gradebook database"""
        connection = sqlite3.connect(self.database_path)  #pylint: disable=E1101
        # This row factory enables accessing row values by column-name
        connection.row_factory = sqlite3.Row   #pylint: disable=E1101
        # Default compiles of sqlite do not enforce foreign key constraints.
        # Since the gradebook schema uses foreign key constraints and the "ON
        # DELETE CASCADE" feature, we ensure constraint enforcement is on.

        # 多表关联
        #connection.execute("PRAGMA foreign_keys=ON")

        return connection

    def close(self):  #pylint: disable=C0116
        self.con.commit()
        self.con.close()

    def execute(self, query, args=None, commit=False):
        """
        Execute a query with the supplied query parameters.
        By default, we don't commit after each call to execute for performance
        reasons. By not committing, sqlite3 automatically creates and commits
        transactions when necessary (like before a SELECT statement).  This is
        very useful when doing many sequential AR saves because one big
        transaction is MUCH faster than many small transactions.
        """
        cur = self.con.cursor()
        # Uncomment the following line to log to the screen the SQL that is executed
        # TODO：研究args作用
        cur.execute(query, args)
        if commit:
            self.con.commit()
        return cur

    def create_table(self, xlsx_path):
        basename = path.basename(xlsx_path).split('.')[0]
        # TODO：sql注入
        df = pd.read_excel(xlsx_path) 
        columns = df.columns
        column_expr = ','.join(columns).strip('\n')
        create_query = f'CREATE TABLE {basename} ({column_expr})'
        self.con.cursor().execute(create_query)
        # self.execute(create_query)
        data = []
        for _ in df.index:
            data.append(tuple(df.loc[_]))
        self.con.cursor().executemany(f'INSERT INTO {basename} VALUES (?, ?, ?, ?, ?, ?)', data)

    def query_data(self, *data_column, **condition):
        data_expr = ' '.join(data_column)
        condition_expr = ''
        for key in condition:
            value = condition[key]
            condition_expr = f'{key}="{value}"'
        # query = f'SELECT * FROM test_data'
        query = f'SELECT {data_expr} FROM test_data WHERE {condition_expr}'
        # 为什么不显示结果？
        for row in self.con.cursor().execute(query):
            print(row)

if __name__ == '__main__':
    excel.to_excel()
    xlsx_path = 'test_data.xlsx'
    db_name = '端子排信息表.db'
    db = Database(db_name)
    # db.query_data('所属安装单位名称', '测试信息', 所属屏柜名称='110kV线路控制屏')
    db.create_table(xlsx_path)
    db.close()

# 对于一个元件，包括文字名称（与原理图相同）和元件代号(所属安装单位名称)
# 端子排是整个屏的端子排，屏内外和元件必须通过端子排连接
# 元件端子的名称包括所属安装单位号，所属元件号及在本元件上的编号，I2-3（第I安装单位第2元件第3端子）
# 端子排视为安装单位的一个特殊元件，I-3为第I安装单位中端子排的第3号端子片
# 对于一个端子，暂定其数据包括：
# -所属屏柜名称：中文
# -所属安装单位名称：中文
# -安装单位号：罗马数字I
# -所属元件号：数字1
# -元件中编号：数字2