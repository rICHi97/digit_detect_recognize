# -*- coding: utf-8 -*-
"""
Created on 2022-03-29 16:50:37

@author: Li Zhi
"""
import os
from os import path

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import applications, layers, optimizers, preprocessing, utils

from . import cfg, east_data, network
from ..recdata import rec

Input = keras.Input
Model = keras.Model
VGG16 = applications.vgg16.VGG16
BatchNormalization = layers.BatchNormalization
Concatenate = layers.Concatenate
Conv2D = layers.Conv2D
Layer = layers.Layer
MaxPooling2D = layers.MaxPooling2D
UpSampling2D = layers.UpSampling2D
Adam = optimizers.Adam
EastData = east_data.EastData
EastPreprocess = east_data.EastPreprocess
Rec = rec.Rec


# TODO：增加一个输出通道，判断是标签或铭牌
# TODO：增加主干网
class EastNet(object):
    """
    EastNet的predict和train依赖一个网络
    不能staticmethod
    ----------
    """
    def __init__(
        self,
        backdone='vgg',
    ):
        assert backdone in ('vgg', 'pva')
        self.input_img = Input(
            name='input_img', shape=(None, None, 3), dtype='float32',
        )
        if backdone == 'vgg':
            vgg16 = VGG16(
                input_tensor=self.input_img, weights='imagenet', include_top=False
            )
            features = [vgg16.get_layer(f'block{i}_pool').output for i in (2, 3, 4, 5, )]
            self._f = {
                1: features[3],
                2: features[2],
                3: features[1],
                4: features[0],
            }
        elif backdone == 'pva':
            pvanet = network.PVAnet(self.input_img)
            features = pvanet.features
            self._f = {
                1: features['f1'],
                2: features['f2'],
                3: features['f3'],
                4: features['f4']
            }
        self.network = self.create_network()

    def _h(self, i):
        assert i in (1, 2, 3, 4, )
        chs = {2: 128, 3: 64, 4: 32}
        if i == 1:
            h = self._f[i]
        else:
            ch = chs[i]
            concat = Concatenate(axis=-1)([self._g(i - 1), self._f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(ch, 1, 1, activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(ch, 3, 1, activation='relu', padding='same')(bn2)
            h = conv_3
        return h

    def _g(self, i):
        assert i in (1, 2, 3, 4, )
        if i == 4:
            bn = BatchNormalization()(self._h(i))
            conv_3 = Conv2D(32, 3, activation='relu', padding='same')(bn)
            g = conv_3
        else:
            g = UpSampling2D((2, 2))(self._h(i))
        return g

    def create_network(self):
        """
        创建east network
        Parameters
        ----------
        Returns
        ----------
        keras.Model
        """
        # features_layers_range = [5, 4, 3, 2]
        # before_output = self._g(4)
        before_output = self._g(4)
        # Layers()
        # 增加class_score用于区分是端子铭牌还是端子编号
        inside_score = Conv2D(1, 1, padding='same', name='inside_score')(before_output)
        classes_score = Conv2D(1, 1, padding='same', name='class_score')(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(before_output)
        east_detect = (
            Concatenate(axis=-1, name='east_detect')(
                [inside_score, classes_score, side_v_code, side_v_coord]
            )
        )

        return Model(inputs=self.input_img, outputs=east_detect)

    def plot(self):
        utils.plot_model(
            self.network,
            to_file='model.png',
            show_shapes=True,
            # show_dtype=True,
            show_layer_names=True,
            dpi=200,
            # expand_nested=True,
            # show_layer_activations=True,
        )

    def train(
        self,
        summary=cfg.summary,
        lr=cfg.lr,
        decay=cfg.decay,
        train_generator=EastData.generator,
        steps_per_epoch=cfg.steps_per_epoch,
        epoch_num=cfg.epoch_num,
        verbose=cfg.train_verbose,
        callbacks=cfg.callbacks,
        val_generator=EastData.generator,
        val_steps=cfg.val_steps,
        save_weights_filepath=cfg.save_weights_filepath,
    ):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        if summary:
            self.network.summary()
        callbacks = [EastData.callbacks(type_) for type_ in callbacks]
        # TODO：研究Adam优化器
        self.network.compile(
            loss=EastData.rec_loss,
            optimizer=Adam(lr, decay),
        )
        # TODO：新版的keras貌似已经在fit中集成了fit_generator功能，有待研究
        self.network.fit_generator(
            generator=train_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=epoch_num,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_generator(is_val=True),
            validation_steps=val_steps,
        )
        self.network.save_weights(save_weights_filepath)

    def load_weights(
        self,
        east_weights_filepath=cfg.east_weights_filepath,
    ):
        self.network.load_weights(east_weights_filepath)
        self.is_load_weights = True

    # TODO：对于尺寸较大的图片，先裁切再predict
    # TODO：terminal_23识别有问题
    # TODO：num_img封装图片为batch检测，研究keras文档api调用说明
    # TODO：支持output txt
    # 2/4修改，输出Rec实例形式
    def predict(
        self,
        east_weights_filepath=cfg.east_weights_filepath,
        img_dir_or_path=cfg.img_dir,
        output_txt=cfg.output_txt,
        output_txt_dir=cfg.output_txt_dir,
        max_predict_img_size=cfg.max_predict_img_size,
        show_predict_img=cfg.show_predict_img,
        predict_img_dir=cfg.predict_img_dir,
        num_img=cfg.num_img,
        pixel_threshold=cfg.pixel_threshold,
    ):
        """
        检测图片中端子及铭牌
        Parameters
        ----------
        img_dir_or_path：待识别img的dir，或单张img路径
        output_txt_dir：存放输出txt文件夹
        num_img：暂为1，符合keras api调用接口
        pixel_threshold：需要nms的像素阈值，越低越慢

        Returns
        ----------
        imgs_recs_xy_list：所有图片的多个rec的四点坐标
        imgs_recs_classes_list：所有图片的rec类别信息
        """
        if not self.is_load_weights:
            self.load_weights()

        if path.isdir(img_dir_or_path):
            img_files = os.listdir(img_dir_or_path)
            img_paths = [path.join(img_dir_or_path, img_file) for img_file in img_files]
        else:
            img_paths = [img_dir_or_path]

        imgs_recs_list, recs_list = [], [] # imgs是配合keras的api调用，把多张img封装成batch，事实只有一张

        for img_path in img_paths:

            img = preprocessing.image.load_img(img_path).convert('RGB')
            d_width, d_height = EastPreprocess.resize_img(img, max_predict_img_size)
            scale_ratio_w, scale_ratio_h = img.width / d_width, img.height / d_height
            img = img.resize((d_width, d_height), Image.BICUBIC)
            array_img = preprocessing.image.img_to_array(img)
            array_img_all = np.zeros((num_img, d_height, d_width, 3))  # 封装多张图片，但在此只封装一张
            array_img_all[0] = array_img
            tf_img = applications.vgg16.preprocess_input(array_img, mode='tf')
            x = np.zeros((num_img, d_height, d_width, 3))
            x[0] = tf_img
            # TODO：明确输出y的shape
            y_pred = self.network.predict(x)
            y = y_pred[0]  # x_height / pixel_size * x_width / pixel_size * 8
            y[:, :, :4] = EastData.sigmoid(y[:, :, :4])
            condition = np.greater_equal(y[:, :, 0], pixel_threshold)
            activation_pixels = np.asarray(condition).nonzero()
            # 12/4：nms中已经修改，以适应predict tensor shape
            # 这张图片对应的所有rec的置信度得分，xy_list，classes
            recs_score, recs_after_nms, recs_classes_list = EastData.nms(
                y, activation_pixels, return_classes=True
            )

            for i, _ in enumerate(zip(recs_score, recs_after_nms)):
                score, xy_list = _[0], _[1]
                if np.amin(score) > 0:
                    xy_list = np.reshape(xy_list, (4, 2))
                    xy_list[:, 0] *= scale_ratio_w
                    xy_list[:, 1] *= scale_ratio_h
                    xy_list = np.reshape(xy_list, (8,)).tolist()
                    rec = Rec(xy_list=xy_list, classes=recs_classes_list[i])
                    recs_list.append(rec)

            imgs_recs_list.append(recs_list)

        return imgs_recs_list
