# -*- coding: utf-8 -*-
"""
Created on 2022-01-30 15:52:53

@author: Li Zhi
在recdata模块，excel_db模块之间加工数据以符合要求
"""
def terminal_info_to_id(cibucle_num, install_unit_num, terminal_num):
	"""
	Parameters
	----------
	cibucle_num：计量柜号，2位PJ固定，1-2位种类，2-4位电压，1位结构类别，3位方案编号（1位英文，2位数字）
		种类：整体式1，分体式仪表2，分体式互感器2H
		电压：
			对于1，35/6~10/0.38kV。35kV，35；6kV，06（猜测）；0.38kV，038；
			对于2，固定为0.1kV，010；
			对于2H，6~35kV。
	install_unit_num：安装单位号，1-2位
	terminal_num：端子编号，1-3位

	Returns
	----------
	terminal_id：‘
	"""
	c, i, t = str(cibucle_num), str(install_unit_num), str(terminal_num)

	# 操作计量柜号
	truncate_index = 3
	if '1' in c[2:4]:
		type_ = '01' # 整体式
	elif '2H' in c[2:4]:
		type_ = '2H' # 分体互感器
		truncate_index = 4
	else:
		type_ = '02' # 分体仪表
	voltage_class = c[truncate_index:-4] 
	assert len(voltage_class) in (2, 3), '计量柜编号电压等级有误' ## 2位或3位，35/06/038/010
	if len(voltage_class) == 2:
		# 转为10V单位，06(kV) -> 600(10V)
		voltage_class = (
			100 * int(voltage_class[1]) if voltage_class[0] == '0' else 100 * int(voltage_class)
		)
	else:
		voltage_class = int(voltage_class[1:])
	# 转为10V，右对齐，左侧补零，宽4位
	voltage_class = f'{voltage_class:0>4}'	
	structure, design = c[-4], c[-3:] # 1位结构类别，3位方案编号
	# 计量柜号，宽10位
	c_num = f'{type_}{voltage_class}{structure}{design}'
	# 安装单位号，右对齐，左侧补0，宽2位
	i_num = f'{i:0>2}'
	# 端子编号，右对齐，左侧补0，宽3位
	t_num = f'{t:0>3}'
	terminal_id = f'{c_num}{i_num}{t_num}'

	return terminal_id

# TODO：键名cfg参数化
def df_row_to_dict(row, model_type):
	"""
	df_row -> dict，调用字典存入db
	Parameters
	----------
	Returns
	----------
	"""
	if model_type == 'Terminal':
		# 将每行转为字典{'terminal_id':, 'loop_number':}
		dict_ = {
			'terminal_id': terminal_info_to_id(row['计量柜号'], row['安装单位号'], row['端子编号']),
			'loop_number': row['连接回路编号']
		}

	return dict_
