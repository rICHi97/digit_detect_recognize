 # -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:24:20 2021

@author: LIZHi

"""
import tensorflow as tf
from tensorflow import keras
from pkgs.east import east_net

EastNet = east_net.EastNet

east = EastNet('vgg')
# keras.utils.plot_model(east.network, 'east_model.png', show_layer_names=True)