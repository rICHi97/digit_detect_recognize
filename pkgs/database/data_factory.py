# -*- coding: utf-8 -*-
"""
Created on 2022-01-30 15:52:53

@author: Li Zhi
在recdata模块，excel_db模块，my_database之间加工数据以符合要求
"""
import pandas as pd

from . import cfg

excel_types = cfg.excel_types


# TODO：从标准计量柜id得到原始id
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
        cubicle_id：统一规范格式后计量柜id
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
        cubicle_id = f'PJ{type_}{voltage_class}{structure}{design}'

        return cubicle_id

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
            return plate_text.replace(' ', '') # 去除空格

        def func3(wiring_terminal):
            wiring_terminal = str(wiring_terminal)
            wiring_terminal = wiring_terminal.replace(' ', '') # 去除空格
            wiring_terminal = wiring_terminal.replace('，', ',') # 中文逗号变为英文逗号
            return wiring_terminal

        # TODO：不确定如何区分相号和元件最后一个字母
        def func4(text_symbol):
            n_text_symbol = []
            n_text_symbol.append(text_symbol[0]) # 同类元件中序号
            n_text_symbol.append(text_symbol[1:3].upper()) # 元件双字母文字符号，TV/PJ
            n_text_symbol.append(text_symbol[-1].lower()) # 相号
            n_text_symbol = ''.join(n_text_symbol)
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

        # 屏内元件以元件id,端子或端子id组成
        def func6(in_cubicle_component):
            _ = str(in_cubicle_component).split('-')
            component_num = _[0]
            n_component_num = f'{component_num:0>4}' #  安装单位+元件序号，103补齐为0103

            # 返回数据包含','为屏内元件，否则为端子
            # 需要补充计量柜id才是元件id或端子id
            if n_component_num[2:4] == '00': # 00说明为端子排元件，此时直接连接端子
                if len(_) != 2:
                    raise Exception(f'连接屏内端子{in_cubicle_component}错误')
                install_unit_num = f'{n_component_num[0:2]}'
                terminal_num = f'{_[1]:0>3}'
                n_in_cubicle_component = f'{install_unit_num}{terminal_num}'

            else: # 说明连接屏内元件
                if len(_) == 2:
                    wiring_terminal = _[1]
                else:
                    wiring_terminal = ''
                n_in_cubicle_component = f'{n_component_num},{wiring_terminal}'

            return n_in_cubicle_component

        def func7(install_unit_num):
            return f'{install_unit_num:0>2}'

        # 标准化计量柜编号
        if df_type == 'Cubicle':

            n_cubiclcle_num = df['num'].apply(func1) # 计量柜编号列
            df['num'] = n_cubiclcle_num

        elif df_type == 'InstallUnit':

            n_cubiclcle_num = df['cubicle_num'].apply(func1) # 在这张sheet中列名为'cubicle_num'
            n_plate_text = df['plate_text'].apply(func2) # 安装单位铭牌文本
            n_install_unit_num = df['num'].apply(func7) # 安装单位编号
            df['cubicle_num'] = n_cubiclcle_num
            df['plate_text'] = n_plate_text
            df['num'] = n_install_unit_num

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

    # TODO：数据正确性检查，或者依赖数据库约束
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
        df_dict, excel_type = excel.df_dict, excel.excel_type
        assert excel_type in excel_types, f'excel_type不能为{excel_type}'

        n_df_dict, model_data_dict = {}, {} # 标准化df字典，模型数据字典
        # data_list中查询包含字段cond的data的return_index项，认定只有一个data满足条件
        cond_data = (
            lambda data_list, cond, return_index: (
                [data for data in data_list if cond in data][0][return_index]
            )
        )
        if excel_type == '二次回路信息表':

            for df_type in ('Cubicle', 'InstallUnit', 'ComponentInfo', 'Component', 'Connection'):
                df = df_dict[df_type]
                n_df = DataFactory.normalize(df, df_type)
                n_df_dict[df_type] = n_df

            # model = Cubicle
            # data = id_, location
            data_df = n_df_dict['Cubicle']
            data_list = data_df.apply(lambda row: (row['num'], row['location']), axis=1).to_list()
            model_data_dict['Cubicle'] = data_list

            # model = InstallUnit
            # model_data = id_, num, plate_text, cubicle
            data_df = n_df_dict['InstallUnit']
            data_list = data_df.apply(
                lambda row: (
                    f"{row['cubicle_num']}{row['num']:>02}",
                    int(row['num']),
                    row['plate_text'],
                    row['cubicle_num'],
                ),
                axis=1,
            ).to_list()
            model_data_dict['InstallUnit'] = data_list

            # model = Component
            # model_data = id_, num, text_symbol, type_, wiring_terminal, install_unit
            data_df = n_df_dict['ComponentInfo']
            TV_row = data_df[data_df['type'].isin(['TV'])]
            PJ_row = data_df[data_df['type'].isin(['PJ'])]
            wiring_terminal_dict = {
                'TV': TV_row['wiring_terminal'].to_list()[0],
                'PJ': PJ_row['wiring_terminal'].to_list()[0],
            }
            data_df = n_df_dict['Component']
            data_list = data_df.apply(
                lambda row: (
                    f"{row['cubicle_num']}{row['install_unit_num']:>02}{row['num']:>02}",
                    int(row['num']),
                    row['text_symbol'],
                    f"{row['text_symbol'][1:-1]}", # row['text_symbol'][1:-1] = component type
                    f"{wiring_terminal_dict[row['text_symbol'][1:-1]]}",
                    f"{row['cubicle_num']}{row['install_unit_num']:>02}",
                ),
                axis=1,
            ).to_list()
            model_data_dict['Component'] = data_list

            iu_list = model_data_dict['InstallUnit']

            # TODO：当未查询到安装单位时报错
            # model = Terminal
            # model_data = id_, num, install_unit
            data_df = n_df_dict['Connection']
            data_list = data_df.apply(
                lambda row: (
                    f"{cond_data(iu_list, row['plate_text'], 0)}{row['terminal_num']:>03}",
                    int(f"{row['terminal_num']}"),
                    f"{cond_data(iu_list, row['plate_text'], 0)}",
                ),
                axis=1,
            ).to_list()
            model_data_dict['Terminal'] = data_list

            # model = Connection
            # model_data = id_, terminal, type_, target
            data_df = n_df_dict['Connection']
            data_list = []
            id_ = 0
            # terminal connect to target
            for row in data_df.itertuples():
                cubicle_num = getattr(row, 'cubicle_num')
                plate_text = getattr(row, 'plate_text')
                install_unit_id = cond_data(iu_list, plate_text, 0)
                terminal_id = f"{install_unit_id}{getattr(row, 'terminal_num'):>03}"
                # 连接类型
                for type_ in (
                    'out_cubicle_component', 'cable', 'out_cubicle_loop', 'in_cubicle_loop'
                ):
                    target = getattr(row, type_)
                    if not pd.isna(target):
                        data = (id_, terminal_id, type_, target)
                        data_list.append(data)
                        id_ += 1
                target = getattr(row, 'in_cubicle_component')
                if not pd.isna(target):
                    # 含','的为元件，否则为端子
                    type_ = 'in_cubicle_component' if ',' in target else 'in_cubicle_terminal'
                    target = f'{cubicle_num}{target}'
                    data = (id_, terminal_id, type_, target)
                    data_list.append(data)
                    id_ += 1
            model_data_dict['Connection'] = data_list

        elif excel_type == '人员信息表':

            for df_type in ('Operator', ):  #pylint: disable=C0325
                df = df_dict[df_type]
                n_df = DataFactory.normalize(df, df_type)
                n_df_dict[df_type] = n_df

            # model = Operator
            # data = name_, tel
            data_df = n_df_dict['Operator']
            data_list = data_df.apply(lambda row: (row['name'], row['tel']), axis=1).to_list()
            id_list = [(i, ) for i in range(len(data_list))]
            data_list = [id_list[i] + data_list[i] for i in range(len(data_list))]
            model_data_dict['Operator'] = data_list

        return model_data_dict

    @staticmethod
    def three_phase_components(all_components):
        # 去除相号a, b, c
        _ = {component[0][:-1] for component in all_components}
        three_phase_components = {
            'TV': [component for component in _ if 'TV' in component],
            'PJ': [component for component in _ if 'PJ' in component],
        }
        return three_phase_components
