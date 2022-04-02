# -*- coding: utf-8 -*-
"""
Created on 2022-03-20 09:36:12

@author: Li Zhi
各模块使用的数据库查询数据通用接口
"""
import peewee

from . import data_factory, my_database

DataFactory = data_factory.DataFactory
Connection = my_database.Connection
Component = my_database.Component
Cubicle = my_database.Cubicle
InstallUnit = my_database.InstallUnit
Operator = my_database.Operator
Terminal = my_database.Terminal


# TODO：取完数据后关闭数据库连接
class DataRetrieve(object):
    """
    数据查询
    """
    @staticmethod
    def cubicle_location(cubicle_id):
        """
        查询指定计量柜对应地理位置
        Parameters
        ----------
        cubicle_id：计量柜id

        Returns
        ----------
        """
        pass

    @staticmethod
    def install_unit_id(cubicle_id, plate_text):
        """
        查询指定计量柜中，指定铭牌文本对应安装单位
        Parameters
        ----------
        cubicle_id：计量柜id，扫描二维码后得到
        plate_text：矫正后的铭牌文本

        Returns
        ----------
        """
        install_unit = (
            InstallUnit
            .select()
            .where(
                (InstallUnit.plate_text == plate_text)
                & (InstallUnit.cubicle == cubicle_id)
            )
            .get()
        )
        return install_unit.id_

    @staticmethod
    def all_operators():
        operators = []
        for _ in Operator.select():
            operators.append((_.name, _.tel))
        return operators


    @staticmethod
    def all_components():
        """
        查询数据库中所有元件
        Parameters
        ----------
        Returns
        ----------
        """
        components = []
        for _ in Component.select():
            components.append((_.text_symbol, ))
        return components
