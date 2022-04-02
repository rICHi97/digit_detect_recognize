# -*- coding: utf-8 -*-
"""
Created on 2022-02-04 11:09:47

@author: Li Zhi
"""
# TODO：定义Group类，描述包括0或1个plate+n个terminal


# TODO：Rec的两个子类Terminal/Plate
# TODO：joint_x_position不应作为Rec属性
# rec.__init__() = Rec.__init__(self)
class Rec(object):

    def __init__(
        self,
        xy_list,
        classes=None,
        text=None,
        joint_x_position=None,
        id_=None,
    ):
        self.xy_list = xy_list
        self.classes = classes
        self.text = text
        self.joint_x_position = joint_x_position
        self.id_ = id_

    def set_attr(self, **kargs):  #pylint: disable=C0116
        for key, value in kargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'{key}不是有效属性')


# class Rec(object):

#     def __init__(
#         self,
#         xy_list,
#         text=None,
#     ):
#         self.xy_list = xy_list
#         self.text = text

# # TODO：joint_x_position不应作为Terminal属性
# class Terminal(Rec):

#     def __init__(
#         self,
#         xy_list,
#         text=None,
#         joint_x_position=None,
#         id_=None,
#     ):
#         Rec.__init__(self, xy_list, text)
#         self.joint_x_position = joint_x_position
#         self.id_ = id_


# class Plate(Rec):

#     def __init__(
#         self,
#         xy_list,
#         text=None,
#         install_unit_id=None,
#     ):
#         Rec.__init__(self, xy_list, text)
#         self.id_ = id_
