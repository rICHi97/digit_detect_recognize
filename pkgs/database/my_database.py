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
# TODO：不能用Model.column_name获取数据，需要研究
class Task(BaseModel):
    """
    检测任务表
    """
    id_ = IntegerField(primary_key=True) # 主码
    type_ = CharField(column_name='type') # 任务类型
    deliver_time = DateTimeField() # 任务下发时间


class Operator(BaseModel):
    """
    操作人员表
    """
    id_ = IntegerField(primary_key=True) # 人员id
    name = CharField()
    tel = IntegerField()


class Cubicle(BaseModel):
    """
    计量柜表
    """
    id_ = FixedCharField(primary_key=True) # 计量柜型号规范化后作为主码id


class InstallUnit(BaseModel):
    """
    安装单位表
    """
    id_ = IntegerField(primary_key=True) # 主码自增
    plate_text = CharField() # 安装单位铭牌文本
    cubicle = ForeignKeyField(Cubicle, backref='install') # 计量柜。同Terminal，1对多


class Terminal(BaseModel):
    """
    端子表
    """
    # 端子id，由计量柜id + 安装单位id + 端子编号得到
    id_ = FixedCharField(primary_key=True)
    num = IntegerField() # 端子编号
    # 安装单位id。1对多，将联系归到多侧，加上1侧主码
    install_unit = ForeignKeyField(InstallUnit, backref='terminal')


class Loop(BaseModel):
    """
    回路表
    """
    id_ = IntegerField(primary_key=True) # 主码自增
    num = CharField() # 回路编号


# TODO：能否继承BaseModel同时重新赋值类属性
class Connection(BaseModel):
    """
    连接关系表，m对n
    """
    id_ = IntegerField(primary_key=True) # 主键
    terminal = ForeignKeyField(Terminal)
    loop = ForeignKeyField(Loop)

    class Meta:
        indexes = (
            (('id_', 'terminal', 'loop'), True),
        )


class TaskOperation(BaseModel):
    """
    检测任务主体、操作人员主体、涉及端子主体联系表
    """
    id_ = IntegerField(primary_key=True) # 主键
    task = ForeignKeyField(Task)
    operator = ForeignKeyField(Operator)
    terminal = ForeignKeyField(Terminal)

    class Meta:
        indexes = (
            (('task', 'operator', 'terminal'), True),
        )

_models = {
    'Task': (Task, ),
    'Operator': (Operator, ),
    'Cubicle': (Cubicle, [Cubicle.id_]),
    'InstallUnit': (InstallUnit, [InstallUnit.id_, InstallUnit.plate_text, InstallUnit.cubicle]),
    'Terminal': (Terminal, [Terminal.id_, Terminal.num, Terminal.install_unit]),
    'Loop': (Loop, [Loop.id_, Loop.num]),
    'Connection': (Connection, [Connection.id_, Connection.terminal, Connection.loop]),
    'TaskOperation': (TaskOperation, ),
}

def create_tables():  #pylint: disable=C0116
    with db:
        tables = [value[0] for value in list(_models.values())]
        db.create_tables(tables)

def connect():  #pylint: disable=C0116
    if db.is_closed():
        db.connect()

def close():  #pylint: disable=C0116
    if not db.is_closed():
        db.close()

# TODO：装饰器
def store(model_data_dict):
    """
    Parameters
    ----------
    model_data_dict：key = model_type_name, value = row_data, see fields in _models.

    Returns
    ----------
    """
    for key, value in model_data_dict.items():
        assert key in _models.keys(), f'model_type不能为{key}'
        model, fields = _models[key][0], _models[key][1]
        connect()
        with db.atomic() as transaction:
            model.insert_many(value, fields).execute()
        close()

if __name__ == '__main__':
    pass
