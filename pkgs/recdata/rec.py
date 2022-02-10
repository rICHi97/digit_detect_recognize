# -*- coding: utf-8 -*-
"""
Created on 2022-02-04 11:09:47

@author: Li Zhi
"""


# TODO：Rec和Recdata可以整合
class Rec(object):

    def __init__(
        self, 
        xy_list, 
        classes=None, 
        text=None, 
        plate_text=None, 
        joint_x_position=None,
        terminal_id=None,
    ):
        self.xy_list = xy_list
        self.classes = classes
        self.text = text
        self.plate_text = plate_text
        self.joint_x_position = joint_x_position
        self.terminal_id = terminal_id

    def set_attr(self, **attrs):
        _keys = ('xy_list', 'classes', 'text', 'plate_text', 'joint_x_position', 'terminal_id')
        for key, value in attrs.items():
            assert key in _keys, f'不存在属性{key}'
            if key == 'xy_list':
                self.xy_list = value
            elif key == 'classes':
                self.classes = value
            elif key == 'text':
                self.text = value
            elif key == 'plate_text':
                self.plate_text = value
            elif key == 'joint_x_position':
                self.joint_x_position = value
            elif key == 'terminal_id':
                self.terminal_id = terminal_id
