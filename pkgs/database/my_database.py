# -*- coding: utf-8 -*-
"""
Created on 2021-11-28 02:37:33

@author: Li Zhi
本模块用于定义数据库及表（Model）
"""
from os import path

import peewee
from . import cfg

CharField = peewee.CharField
DateField = peewee.DateField
DateTimeField = peewee.DateTimeField
FixedCharField = peewee.FixedCharField
SmallIntegerField = peewee.SmallIntegerField

Model = peewee.Model # Model对应表


# TODO：st sql插件支持
class MyDatabase(object):
    """
    用于创建、维持数据库，并提供命令执行。
    本类基于一个给定数据库db文件进行操作
    """
    # TODO：db存在就删除
    def __init__(
        self,
        database_path=cfg.database_path,
        pragmas=cfg.pragmas
    ):
        """
        在指定位置创建数据库
        Parameters
        ----------
        Returns
        ----------
        """
        self.database_path = database_path
        self.database = peewee.SqliteDatabase(self.database_path, pragmas)

db = MyDatabase().database # 测试，后续可能需要修改


# TODO:能否通过api设置database，而不是固定在定义中；见doc_setting the database at run-time
class BaseModel(Model):
    class Meta:
        database = db


# TODO：考虑主键问题
# TODO：研究table_name
class Inspection(BaseModel):
    """
    检测任务表
    """
    task_id = CharField()
    task_type = CharField()
    task_deliver_time = DateField()


class Terminal(BaseModel):
    """
    端子表
    """
    terminal_id = FixedCharField()
    loop_number = CharField()                                                           # 端子连接回路编号

    def get_loop_number(query_terminal_id):
        loop_number = Terminal.get(Terminal.terminal_id == query_terminal_id).loop_number
        return loop_number

def create_tables():
    db.connect()
    db.create_tables([Inspection, Terminal])

def connect_db():
    db.connect()

def close_db():
    db.close()

def store(model_type, **query):
    """
    Parameters
    ----------
    model_type：'Inspection'/'Terminal'
    query：column=value

    Returns
    ----------
    """
    if model_type == 'Inspection':
        Inspection.create(
            task_id=query['task_id'], 
            task_type=query['task_type'],
            task_deliver_time=query['task_deliver_time']
        )
    elif model_type == 'Terminal':
        Terminal.create(
            terminal_id=query['terminal_id'],
            loop_number=query['loop_number'],
        )

if __name__ == '__main__':
    pass