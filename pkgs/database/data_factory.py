# -*- coding: utf-8 -*-
"""
Created on 2022-01-30 15:52:53

@author: Li Zhi
在recdata模块，excel_db模块，my_database之间加工数据以符合要求
"""
import pandas as pd

from . import cfg

excel_types = cfg.excel_types


class DataFactory():
    """
    主要处理数据本身
    """
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

    @staticmethod
    def get_terminal_id(cubicle_id, install_unit_id, terminal_num):
        """
        Parameters
        ----------
        cubicle_id：规范化后的计量柜id
        install_unit_num：安装单位号，1-2位
        terminal_num：端子编号，1-3位

        Returns
        ----------
        terminal_id：
        """
        c, i, t = str(cubicle_id), str(install_unit_id), str(terminal_num)
        # 安装单位号，右对齐，左侧补0，宽2位
        i = f'{i:0>2}'
        # 端子编号，右对齐，左侧补0，宽3位
        t = f'{t:0>3}'
        terminal_id = f'{c}{i}{t}'

        return terminal_id

    @staticmethod
    def normalize(row, excel_type):
        """
        规范标准化原始表格的数据，作为表格和数据库之间的中介
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
        elif excel_type == '检测任务':
            keys = []
        elif excel_type == '人员信息':
            keys = []

        for key in keys:
            column_name = column_names[key]
            d[key] = row[column_name]

        s = pd.Series(d)

        return s

    @staticmethod
    def create_data(n_df, excel_type):
        """
        创建直接用于插入数据库的数据。
        难点是data需要和插入数据中Field顺序一致
        Parameters
        ----------
        n_df：规范标准化后的表格df

        Returns
        ----------
        model_data_dict：
            dict，key = model_type，value = [(fields_data)]
        """
        assert excel_type in excel_types, f'excel_type不能为{excel_type}'
        model_data_dict = {}

        if excel_type == '端子信息':

            temp_connections = []
            cubicles, install_units, terminals, loops, connections = [], [], [], [], []

            c_ids = n_df['cubicle_id'].unique() # 多条数据指向同一个计量柜
            i_id = 0
            for c_id in c_ids:
                # 计量柜
                c_data = (c_id, ) # C.id_
                cubicles.append(c_data)

                # 多条数据指向同一个计量柜中的同一个安装单位
                conditon = (n_df['cubicle_id'] == c_id)
                p_texts = n_df[conditon]['plate_text'].unique()
                # 手动设置主键为i，即数据次序，方便设置外键
                for p_text in p_texts:
                    i_data = (i_id, p_text, c_id) # I.id_, I.p_text, I.c_id，通过id连接外键
                    install_units.append(i_data)

                    # 多条数据指向同一个计量柜中的同一个安装单位中的同一个端子
                    conditon = (n_df['cubicle_id'] == c_id) & (n_df['plate_text'] == p_text)
                    t_nums = n_df[conditon]['terminal_num'].unique()
                    for t_num in t_nums:
                        t_num = int(t_num)
                        t_id = DataFactory.get_terminal_id(c_id, i_id, t_num)
                        t_data = (t_id, t_num, i_id) # T.id_, T.num, T.i_id
                        terminals.append(t_data)

                        # 查询这个端子对应的连接回路
                        conditon = (conditon) & (n_df['terminal_num'] == t_num)
                        connected_loop_nums = n_df[conditon]['loop_num']
                        for connected_loop_num in connected_loop_nums:
                            # 存储端子id和回路文本
                            connection = (t_id, connected_loop_num)
                            temp_connections.append(connection)
                    i_id += 1

            # 端子连接回路是一个难点，因为是多对多
            # 逐个检查连接，同时生成loops和connections
            l_id, num2id = 0, {}
            for i, connection in enumerate(temp_connections):
                t_id, loop_num = connection[0], connection[1]

                # 此时是一条新的回路
                if loop_num not in num2id.keys():
                    l_data = (l_id, loop_num) # L.id_, L.num
                    loops.append(l_data)
                    l_id += 1
                    num2id[loop_num] = l_id

                # 无论新老回路，这个连接都是连接表中的一条有效数据
                # 不能直接插入l_id
                # 举例：三条连接分别是回路1、2、1；第3条连接验证后，l_id是2，但该条连接回路1对应id应是1
                # i是连接的次序作为主键，查询该条回路对应次序作为回路外键，该条连接中的t_id作为端子外键
                # Connection.id,  Connection.terminal_id, Connection.loop_id
                connection_data = (i, t_id, num2id[loop_num])
                connections.append(connection_data)

            # 数据顺序不能颠倒，因为有外键关系
            model_data_dict = {
                'Cubicle': cubicles,
                'InstallUnit': install_units,
                'Terminal': terminals,
                'Loop': loops,
                'Connection': connections
            }

        return model_data_dict
