# -*- coding: utf-8 -*-
"""
Created on 2022-02-04 11:09:47

@author: Li Zhi
"""


# TODO：Rec和Recdata可以整合
class Rec(object):

    def __init__(self, xy_list, classes=None, text=None, group=None, joint_x_position=None):
        self.xy_list = xy_list
        self.classes = classes
        self.text = text
        self.group = group
        self.joint_x_position = joint_x_position

    def set_attr(self, **attrs):
        for key, value in attrs.items():
            assert key in ('xy_list', 'classes', 'text', 'group', 'joint_x_position'), (
                f'不存在属性{key}'
            )
            if key == 'xy_list':
                self.xy_list = value
            elif key == 'classes':
                self.classes = value
            elif key == 'text':
                self.text = value
            elif key == 'group':
                self.group = value
            elif key == 'joint_x_position':
                self.joint_x_position = value
