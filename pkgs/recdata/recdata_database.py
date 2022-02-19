# -*- coding: utf-8 -*-
"""
Created on 2022-02-07 18:49:46

@author: Li Zhi
"""
import peewee

from ..database import my_database

Connection = my_database.Connection
InstallUnit = my_database.InstallUnit
Terminal = my_database.Terminal

# TODO：研究join
class RecdataDB(object):
    """
    封装端子查询
    """
    def __init__(self, cubicle_id):
        self.cubicle_id = cubicle_id

    def get_install_unit_id(self, plate_text):
        """
        Parameters
        ----------
        plate_text：矫正后的铭牌文本

        Returns
        ----------
        install_unit_id：安装单位id
        """
        try:
            install_unit = (
                InstallUnit
                .select()
                .where(
                    (InstallUnit.plate_text == plate_text)
                    & (InstallUnit.cubicle == self.cubicle_id)
                )
                .get()
            )
            return install_unit.id_
        except InstallUnit.DoesNotExist:  #pylint: disable=E1101
            print('不存在')

    def get_terminal_connected_loop(self, terminal_id):
        """
        Parameters
        ----------
        terminal_id：端子id

        Returns
        ----------
        """
        connections = (
            Connection
            .select()
            .where(Connection.terminal == terminal_id)
        )
        for connection in connections:
            print(connection.loop.num)
