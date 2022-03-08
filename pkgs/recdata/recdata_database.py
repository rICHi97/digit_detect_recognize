# -*- coding: utf-8 -*-
"""
Created on 2022-02-07 18:49:46

@author: Li Zhi
"""
import peewee

from ..database import data_factory, my_database

DataFactory = data_factory.DataFactory
Connection = my_database.Connection
InstallUnit = my_database.InstallUnit
Loop = my_database.Loop
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
            print(f'{plate_text}不存在')

    # TODO：返回值优化，避免直接修改group_list？
    def get_terminals_id(self, group_list):
        """
        Parameters
        ----------
        group_list：元素为列表，列表第一个元素为安装单位铭牌，其余为该安装单位中端子。或者都为端子，无铭牌

        Returns
        ----------
        """
        for group in group_list:
            if not group[0].classes == 'plate':
                print('本组无铭牌')
                # ui中提示并可手动输入
                terminal_id = None
                continue
            plate_text = group[0].text
            install_unit_id = self.get_install_unit_id(plate_text)
            for i, rec in enumerate(group):
                if i == 0:
                    rec.id_ = install_unit_id
                terminal_num = group[i].text
                terminal_id = DataFactory.get_terminal_id(
                    self.cubicle_id, install_unit_id, terminal_num
                )
                rec.id_ = terminal_id

        return group_list

    @staticmethod
    def get_connected_loops_id(terminal_id):
        """
        Parameters
        ----------
        terminal_id：端子id

        Returns
        ----------
        """
        loops_id = []
        connections = (
            Connection
            .select()
            .where(Connection.terminal == terminal_id)
        )
        for connection in connections:
            loops_id.append(connection.loop.id_)

        return loops_id

    @staticmethod
    def get_loop_num(loop_id):
        """
        Parameters
        ----------
        loop_id：回路id

        Returns
        ----------
        """
        loop = (
            Loop
            .select()
            .where(Loop.id_ == loop_id)
            .get()
        )
        loop_num = loop.num
        return loop_num

    def find_terminal(self, group_list, *loops_num):
        pass