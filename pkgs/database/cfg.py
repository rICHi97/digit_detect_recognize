# -*- coding: utf-8 -*-
"""
Created on 2022-01-23 14:13:25

@author: Li Zhi
"""
# excel_db_io
excel_paths = {
	'Terminal': './resource/test_data/excel/端子信息表.xlsx',
}

# my_database
database_path = './resource/database/on-site_inspection.db'
pragmas = {
	'journal_mode': 'wal',
	'foreign_keys': 1, # 开启外码
	'ignore_check_constraints': 0,
}
