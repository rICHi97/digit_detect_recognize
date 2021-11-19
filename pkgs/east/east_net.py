# -*- coding: utf-8 -*-
"""
Created on 2021-11-19 00:36:11

@author: Li Zhi
"""

"""
本模块用以进行east网络的搭建及训练工作
"""
import os

from keras import  callbacks, optimizers
import tensorflow

EarlyStopping = callbacks.EarlyStopping
LearningRateScheduler = callbacks.LearningRateScheduler
ModelCheckpoint = callbacks.ModelCheckpoint
ReduceLRonPlateau = callbacks.ReduceLROnPlateau
TensorBoard =  callbacks.TensorBoard
Adam = optimizers.Adam
Session = tensorflow.Session
logging = tensorflow.logging
ConfigProto = tensorflow.ConfigProto


class EastNet(object):
	"""
	Parameters
	----------
	
	Returns
	----------
	"""
	@staticmethod
	def network():
		pass

	@staticmethod
	def train():
		pass