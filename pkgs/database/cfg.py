# -*- coding: utf-8 -*-
"""
Created on 2022-01-23 14:13:25

@author: Li Zhi
"""
# my_database
database_path = './resource/database/on-site_inspection.db'
pragmas = {
	'journal_mode': 'wal',
	'foreign_keys': 1, # 开启外码
	'ignore_check_constraints': 0,
}

excel_types = ['二次回路信息表', '检测任务', '人员信息表']
# excel_db_io，data_factory
# 表格路径
excel_paths = {
	'二次回路信息表': './resource/test_data/excel/二次回路信息表.xlsx',
	'人员信息表': './resource/test_data/excel/人员信息表.xlsx',
}
# 表格参数
# {
# 	表格类型:{
# 		工作表类型:{
# 			工作表名,
# 			数据列名
# 		}
# 	}
# }
# 表格中的数据需要加工后送入数据库
# 元件信息 + 安装单位元件 = db.元件
# 连接 = db.端子, db.连接
excel_args = {
	'二次回路信息表': {
		'Cubicle': {
			'sheet_name': '计量柜信息表',
			'num': '计量柜编号',
			'location': '计量柜位置',
			'info': '计量柜信息',
		},
		'InstallUnit': {
			'sheet_name': '安装单位表',
			'cubicle_num': '计量柜编号',
			'num': '安装单位序号',
			'plate_text': '安装单位铭牌文本',
		},
		'ComponentInfo': {
			'sheet_name': '元件信息表',
			'type': '元件类型',
			'wiring_terminal': '所属接线端子',
		},
		'Component': {
			'sheet_name': '安装单位元件表',
			'cubicle_num': '计量柜编号',
			'install_unit_num': '安装单位序号',
			'num': '元件序号',
			'text_symbol': '元件文字符号',
		},
		'Connection': {
			'sheet_name': '端子连接表',
			'cubicle_num': '计量柜编号',
			'plate_text': '安装单位铭牌文本',
			'out_cubicle_component': '屏外元件-端子',
			'cable': '电缆编号',
			'out_cubicle_loop': '屏外回路编号',
			'terminal_num': '端子编号',
			'in_cubicle_loop': '屏内回路编号',
			'in_cubicle_component': '屏内元件-端子',
		}
	},
	'人员信息表': {
		'Operator': {
			'sheet_name': '人员信息表',
			'name': '检验人员姓名',
			'tel': '联系方式',
		}
	},
}
