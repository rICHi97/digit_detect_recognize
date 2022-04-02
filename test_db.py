# -*- coding: utf-8 -*-
"""
Created on 2022-01-28 18:48:57

@author: Li Zhi
"""
import time

from pkgs.database import excel_db_io, my_database
from pkgs.recdata import recdata_database
# TODO：会被别的程序占用
excel_db_io.excel2db()
# RecdataDB = recdata_database.RecdataDB

# r_db = RecdataDB('013500AG23')
# r_db.get_terminal_connected_loop('020100AG2302020')