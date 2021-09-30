# -*- coding: utf-8 -*-
"""
Created on 2021-09-11 15:24:57

@author: Li Zhi
"""


class UserException(Exception):

	def __init__(self, error_msg):

		self.error_msg = error_msg