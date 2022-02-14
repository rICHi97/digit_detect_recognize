# -*- coding: utf-8 -*-
"""
Created on 2022-01-23 14:13:25

@author: Li Zhi
"""
excel_types = ['端子信息', '检测任务', '人员信息']

# data_factory
# 表格列名
excel_column_names = {
	'cubicle_id': '计量柜号',
	'plate_text': '安装单位铭牌文本',
	'terminal_num': '端子编号',
	'loop_num': '连接回路编号',
}

# excel_db_io
excel_paths = {
	'端子信息': './resource/test_data/excel/端子信息表.xlsx',
}

# my_database
database_path = './resource/database/on-site_inspection.db'
pragmas = {
	'journal_mode': 'wal',
	'foreign_keys': 1, # 开启外码
	'ignore_check_constraints': 0,
}
