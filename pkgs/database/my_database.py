# -*- coding: utf-8 -*-
"""
Created on 2021-11-28 02:37:33

@author: Li Zhi
本模块用于定义数据库及表（Model）
"""
from os import path

import peewee
from . import cfg

AutoField = peewee.AutoField
CharField = peewee.CharField
CompositeKey = peewee.CompositeKey
DateTimeField = peewee.DateTimeField
FixedCharField = peewee.FixedCharField
ForeignKeyField = peewee.ForeignKeyField
IntegerField = peewee.IntegerField

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


# TODO：研究table_name
# TODO：严禁Field.column_name属性，id/type都是python关键字，直接作为类属性命名不方便
class Task(BaseModel):
    """
    检测任务表
    """
    task_id = IntegerField(primary_key=True) # 主码
    task_type = CharField() # 任务类型
    task_deliver_time = DateTimeField() # 任务下发时间


class Operator(BaseModel):
    """
    操作人员表
    """
    operator_id = IntegerField(primary_key=True) # 人员id
    operator_name = CharField()
    operator_tel = IntegerField()


class Cibucle(BaseModel):
    """
    计量柜表
    """
    cubicle_id = AutoField() # 主码自增   


class InstallUnit(BaseModel):
    """
    安装单位表
    """
    install_unit_id = AutoField() # 主码自增
    plate_text = CharField() # 安装单位铭牌文本
    cubicle_id = ForeignKeyField(Cubicle) # 计量柜id。同Terminal，1对多


class Terminal(BaseModel):
    """
    端子表
    """
    terminal_id = FixedCharField(primary_key=True) # 端子id，由安装单位id + 端子编号得到
    terminal_num = IntegerField() # 端子编号
    install_unit_id = ForeignKeyField(InstallUnit) # 安装单位id。1对多，将联系归到多侧，加上1侧主码


class Loop(BaseModel):
    """
    回路表
    """
    loop_id = AutoField() # 主码自增
    loop_num = CharField() # 回路编号


# TODO：能否继承BaseModel同时重新赋值类属性
class Connection(Model):
    """
    连接关系表，m对n
    """
    loop = ForeignKeyField(Loop)
    terminal = ForeignKeyField(Terminal)

    class Meta:
        database = db
        primary_key = CompositeKey('loop', 'terminal')


class TaskOperation(Model):
    """
    检测任务主体、操作人员主体、涉及端子主体联系表
    """
    task = ForeignKeyField(Task)
    operator = ForeignKeyField(Operator)
    terminal = ForeignKeyField(Terminal)

    class Meta:
        database = db
        primary_key = CompositeKey('task', 'operator', 'terminal_id')


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