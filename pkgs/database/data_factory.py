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
    def normalize_cubicle_num(cubicle_num):
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
        c_n = cubicle_num
        # 操作计量柜号，前两位固定为'PJ'
        truncate_index = 3
        if '1' in c_n[2:4]:
            type_ = '01' # 整体式
        elif '2H' in c_n[2:4]:
            type_ = '2H' # 分体互感器
            truncate_index = 4
        else:
            type_ = '02' # 分体仪表
        voltage_class = c_n[truncate_index:-4]
        assert len(voltage_class) in (2, 3), '计量柜编号电压等级有误' # 2位或3位，35/06/038/010
        if len(voltage_class) == 2:
            # 转为10V单位，06(kV) -> 600(10V)，35(kV) -> 3500(10V)
            voltage_class = (
                100 * int(voltage_class[1]) if voltage_class[0] == '0' else 100 * int(voltage_class)
            )
        else:
            # 038(kV) -> 38(10V)
            voltage_class = int(voltage_class[1:])
        # 转为10V，右对齐，左侧补零，宽4位, 3500/0600/0038/0010
        voltage_class = f'{voltage_class:0>4}'
        structure, design = c_n[-4], c_n[-3:] # 1位结构类别，3位方案编号
        # 计量柜号，宽10位
        normative_cubicle_num = f'{type_}{voltage_class}{structure}{design}'

        return normative_cubicle_num

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
    def normalize(df, df_type):
        """
        规范标准化原始表格的数据，作为表格和数据库之间的中介
        Parameters
        ----------
        Returns
        ----------
        s：Series
            端子信息：pd.Series([cubicle_id, plate_text, terminal_num, loop_num])
        """
        def func1(cubicle_num):
            return DataFactory.normalize_cubicle_num(cubicle_num)

        def func2(plate_text):
            return ''.join(plate_text.strip()) # 去除空格

        def func3(wiring_terminal):
            wiring_terminal = str(wiring_terminal)
            wiring_terminal = wiring_terminal.replace(' ', '') # 去除空格
            wiring_terminal = wiring_terminal.replace('，', ',') # 中文逗号变为英文逗号
            return wiring_terminal

        def func4(text_symbol):
            n_text_symbol = []
            n_text_symbol.append(text_symbol[0]) # 同类元件中序号
            n_text_symbol.append(text_symbol[1:3].upper()) # 元件双字母文字符号，TV/PJ
            n_text_symbol.append(text_symbol[-1].lower()) # 相号
            ''.join(n_text_symbol)
            return n_text_symbol

        # 屏外元件以文字符号表示
        def func5(out_cubicle_component):
            _ = out_cubicle_component.split('-')
            text_symbol = _[0]
            n_text_symbol = func4(text_symbol)
            if len(_) == 2:
                wiring_terminal = _[1]
                n_out_cubicle_component = f'{n_text_symbol},{wiring_terminal}'
            else:
                n_out_cubicle_component = f'{n_text_symbol},'
            return n_out_cubicle_component

        # 屏内元件以安装单位+元件序号+接线端子号组成
        def func6(in_cubicle_component):
            _ = in_cubicle_component.split('-')
            component_num = _[0]
            n_component_num = f'{component_num:0>4}' #  安装单位+元件序号，103补齐为0103
            if len(_) == 2:
                wiring_terminal = _[1]
                n_in_cubicle_component = f'{n_component_num},{wiring_terminal}'
            else:
                n_in_cubicle_component = f'{n_component_num},'
            return n_in_cubicle_component

        # 标准化计量柜编号
        if df_type == 'Cubicle':

            n_cubiclcle_num = df['num'].apply(func1) # 计量柜编号列
            df['num'] = n_cubiclcle_num

        elif df_type == 'InstallUnit':

            n_cubiclcle_num = df['cubicle_num'].apply(func1) # 在这张sheet中列名为'cubicle_num'
            n_plate_text = df['plate_text'].apply(func2) # 安装单位铭牌文本
            df['cubicle_num'] = n_cubiclcle_num
            df['plate_text'] = n_plate_text

        elif df_type == 'ComponentInfo':
            n_wiring_terminal = df['wiring_terminal'].apply(func3)
            df['wiring_terminal'] = n_wiring_terminal

        elif df_type == 'Component':
            n_cubiclcle_num = df['cubicle_num'].apply(func1) # 在这张sheet中列名为'cubicle_num'
            n_text_symbol = df['text_symbol'].apply(func4)
            df['cubicle_num'] = n_cubiclcle_num
            df['text_symbol'] = n_text_symbol

        elif df_type == 'Connection':
            n_cubiclcle_num = df['cubicle_num'].apply(func1) # 在这张sheet中列名为'cubicle_num'
            n_plate_text = df['plate_text'].apply(func2) # 安装单位铭牌文本
            n_out_cubicle_component = df['out_cubicle_component'].apply(func5) # 屏外元件及端子
            n_in_cubicle_component = df['in_cubicle_component'].apply(func6) # 屏内元件及端子
            df['cubicle_num'] = n_cubiclcle_num
            df['plate_text'] = n_plate_text
            df['out_cubicle_component'] = n_out_cubicle_component
            df['in_cubicle_component'] = n_in_cubicle_component

        n_df = df

        return n_df

    @staticmethod
    def create_data(excel):
        """
        创建直接用于插入数据库的数据。
        难点是data需要和插入数据中Field顺序一致
        Parameters
        ----------
        excel：excel_db_io.Excel

        Returns
        ----------
        model_data_dict：
            dict，key = model_type，value = [(fields_data)]
        """
        n_df_dict = {} # 标准化df字典
        df_dict, excel_type = excel.df_dict, excel.excel_type

        assert excel_type in excel_types, f'excel_type不能为{excel_type}'

        if excel_type == '二次回路信息表':

            for df_type in ('Cubicle', 'InstallUnit', 'ComponentInfo', 'Component', 'Connection'):
                df = df_dict[df_type]
                n_df = DataFactory.normalize(df, df_type)

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
