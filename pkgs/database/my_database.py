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

IntegrityError = peewee.IntegrityError

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
    tel = CharField()
    password = IntegerField(null=True) # 初次登录时设置密码


class Cubicle(BaseModel):
    """
    计量柜表
    """
    id_ = FixedCharField(primary_key=True) # 计量柜型号规范化后作为主码id，长10位
    location = CharField() # 位置信息


class InstallUnit(BaseModel):
    """
    安装单位表
    """
    id_ = FixedCharField(primary_key=True) # 计量柜id + 2位安装单位序号 = 安装单位id，长12位
    num = IntegerField() # 安装单位序号
    plate_text = CharField() # 安装单位铭牌文本
    cubicle = ForeignKeyField(Cubicle, backref='install_units') # 计量柜。1计量柜对多安装单位


class Component(BaseModel):
    """
    元件表，假定互感器三相二次绕组，每相为一个元件，TVa, TVb, TVc，暂不考虑中性点，暂不考虑不同型号
    规定两类元件：电压互感器TV，电能表PJ，文字符号格式规定为元件同类序号+ 元件双字符 + 相号
    规定都为三相元件，每一相的接线端子都一致，TVa, TVb, TVc都属于TV类型元件
    """
    id_ = FixedCharField(primary_key=True) # 安装单位id + 2位元件序号 = 元件id，长14位
    num = IntegerField() # 元件在安装单位中的序号
    text_symbol = CharField() # 元件文字符号
    type_ = CharField() # 元件类型，主要是电压互感器器和电表，TV/DB
    wiring_terminal = CharField() # 元件的接线端子，区别与端子排中端子
    install_unit = ForeignKeyField(InstallUnit, backref='components') # 安装单位。1安装单位对多元件


class Terminal(BaseModel):
    """
    端子表
    """
    id_ = FixedCharField(primary_key=True) # 端子id，安装单位id + 3位端子编号得到，长15位
    num = IntegerField() # 端子编号
    install_unit = ForeignKeyField(InstallUnit, backref='terminals') # 安装单位id。1对多，将联系归到多侧，加上1侧主码


class Connection(BaseModel):
    """
    连接关系表，1个端子 连接 n个屏外元件-端子/电缆编号/屏外回路编号/屏内元件-端子
    """
    # 每一条数据为连接元组，(端子，屏外元件-端子/电缆编号/屏外回路编号/屏内元件-端子)
    id_ = IntegerField(primary_key=True) # 主键
    terminal = ForeignKeyField(Terminal, backref='connections') # 不可为空，这要求excel中计量柜编号、安装单位文本、端子编号不为空
    type_ = CharField() # 连接类型，分为连接屏外元件-端子/连接电缆/连接屏外回路/连接屏内回路/连接屏内元件-端子/连接屏内端子六类

    # 将六类连接中的端子连接目标归为一个字段，格式规定：
    # 连接屏外元件-端子，out_cubicle_component：'1TVa,3'，端子可以为空
    # 连接电缆，cable：'KVV-4X1.5'
    # 连接屏外/屏内回路，out_cubicle_loop，in_cubicle_loop：'A411'
    # 连接屏内元件-端子，in_cubicle_component：元件id+元件接线端子，'元件id, 2位接线端子'，端子可以为空
    # 连接屏内端子，in_cubicle_terminal：端子id
    target = CharField()

    class Meta:  #pylint: disable=R0903,C0115
        indexes = (
            (('id_', 'terminal', 'type_'), True),
        )


# TODO：设置backref
class TaskOperation(BaseModel):
    """
    检测任务主体、操作人员主体、涉及端子主体联系表
    """
    id_ = IntegerField(primary_key=True) # 主键
    task = ForeignKeyField(Task)
    operator = ForeignKeyField(Operator)
    terminal = ForeignKeyField(Terminal)

    class Meta:  #pylint: disable=R0903,C0115
        indexes = (
            (('task', 'operator', 'terminal'), True),
        )

_models = {
    'Task': (Task, [Task.id_, Task.type_, Task.deliver_time]),
    'Operator': (Operator, [Operator.id_, Operator.name, Operator.tel]),
    'Cubicle': (Cubicle, [Cubicle.id_, Cubicle.location]),
    'InstallUnit': (
        InstallUnit,
        [InstallUnit.id_, InstallUnit.num, InstallUnit.plate_text, InstallUnit.cubicle],
    ),
    'Component': (
        Component,
        [
            Component.id_,
            Component.num,
            Component.text_symbol,
            Component.type_,
            Component.wiring_terminal,
            Component.install_unit
        ]
    ),
    'Terminal': (Terminal, [Terminal.id_, Terminal.num, Terminal.install_unit]),
    'Connection': (
        Connection,
        [Connection.id_, Connection.terminal, Connection.type_, Connection.target]
    ),
    'TaskOperation': (
        TaskOperation,
        TaskOperation.id_, TaskOperation.task, TaskOperation.operator, TaskOperation.terminal
    ),
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
    with db.atomic() as transaction:
        for key, value in model_data_dict.items():
            model, fields = _models[key][0], _models[key][1]
            try:
                model.insert_many(value, fields).execute()
            except IntegrityError as inst:
                print(f'{key}IntegrityError')
                print(inst)
                break
    close()
    return model_data_dict

if __name__ == '__main__':
    pass
