# -*- coding: utf-8 -*-
"""
Created on 2021-11-19 00:36:11

@author: Li Zhi
"""

"""
本模块用以进行east网络的搭建及训练工作
"""
import os
from os import path

from keras import  applications, layers, optimizers, preprocessing, Input, Model
import numpy as np
from PIL import Image
from tensorflow.compat import v1

from . import cfg
from . import east_data
from ..recdata import recdata_io

VGG16 = applications.vgg16.VGG16
preprocess_input = applications.vgg16.preprocess_input
regularizers = layers.regularizers
BatchNormalization = layers.BatchNormalization
Concatenate = layers.Concatenate
Conv2D = layers.Conv2D
UpSampling2D = layers.UpSampling2D
# image = preprocessing.image  容易混淆
Adam = optimizers.Adam
Session = v1.Session
logging = v1.logging
ConfigProto = v1.ConfigProto

EastData = east_data.EastData
EastPreprocess = east_data.EastPreprocess
RecdataIO = recdata_io.RecdataIO


def _init_environ():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  设置多显卡
    # 只显示error
    os.environ['TF_MIN_CPP_LOG_LEVEL'] = '2'
    logging.set_verbosity(logging.ERROR)
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  #pylint: disable=E1101
    config.gpu_options.allow_growth = True  #pylint: disable=E1101
    session = Session(config=config)


class EastNet(object):
    """
    EastNet的predict和train依赖一个网络
    不能staticmethod
    ----------
    """
    def __init__(
        self,
        locked_layers=cfg.locked_layers,
        feature_layers_range=cfg.feature_layers_range,
        feature_layers_num=cfg.feature_layers_num,
    ):
        self.feature_layers_range = feature_layers_range
        self.feature_layers_num = feature_layers_num
        self.input_img = Input(
            name='input_img', shape=(None, None, cfg.num_channels), dtype='float32'
        )

        vgg16 = VGG16(
            input_tensor=self.input_img, weights='imagenet', include_top=False
        )

        if locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'), vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False

        self._f = [vgg16.get_layer(f'block{i}_pool').output for i in feature_layers_range]
        self._f.insert(0, None)
        self.diff = feature_layers_range[0] - feature_layers_num
        self.east_model = self.network()

    def _g(self, i):
        # i+diff in cfg.feature_layers_range
        if not i + self.diff in self.feature_layers_range:
            print(f'i={i}+diff={self.diff} not in {str(self.feature_layers_range)}')
        if i == self.feature_layers_num:
            bn = BatchNormalization()(self._h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)

        return UpSampling2D((2, 2))(self._h(i))

    def _h(self, i):
        # i+diff in cfg.feature_layers_range
        if not i + self.diff in self.feature_layers_range:
            print(f'i={i}+diff={self.diff} not in {str(self.feature_layers_range)}')
        if i == 1:
            return self._f[i]

        concat = Concatenate(axis=-1)([self._g(i - 1), self._f[i]])
        bn1 = BatchNormalization()(concat)
        conv_1 = Conv2D(128 // 2 ** (i - 2), 1, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv_1)
        conv_3 = Conv2D(128 // 2 ** (i - 2), 3, activation='relu', padding='same')(bn2)

        return conv_3

    def network(self):
        """
        创建east network
        Parameters
        ----------

        Returns
        ----------
        keras.Model
        """
        before_output = self._g(self.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score')(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(before_output)
        east_detect = (
            Concatenate(axis=-1, name='east_detect')([inside_score, side_v_code, side_v_coord])
        )

        return Model(inputs=self.input_img, outputs=east_detect)  

    def train(
        self,
        summary=cfg.summary,
        lr=cfg.lr,
        decay=cfg.decay,
        train_generator=EastData.generator,
        steps_per_epoch=cfg.steps_per_epoch,
        epoch_num=cfg.epoch_num,
        verbose=cfg.train_verbose,
        callbacks=None,
        val_generator=EastData.generator,
        val_steps=cfg.val_steps,
        save_weights_dir=cfg.save_weights_dir,
    ):
        """
        Parameters
        ----------

        Returns
        ----------
        """
        if summary:
            self.east_model.summary()
        # TODO：研究Adam优化器
        self.east_model.compile(
            loss=EastData.rec_loss,
            optimizer=Adam(lr, decay),
        )
        # TODO：新版的keras貌似已经在fit中集成了fit_generator功能，有待研究
        self.east_model.fit_generator(
            train_generator(),
            steps_per_epoch,
            epoch_num,
            verbose,
            callbacks,
            val_generator(is_val=True),
            val_steps,
        )
        self.east_model.save_weights(save_weights_dir)

    # TODO：对于尺寸较大的图片，先裁切再predict
    # TODO：支持对单张图片predict
    # TODO：terminal_23识别有问题
    def predict(
        self,
        east_weights_file_path=cfg.east_weights_file_path,
        img_dir=cfg.img_dir,
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

        Returns
        ----------

        """
        self.east_model.load_weights(east_weights_file_path)

        img_files = os.listdir(img_dir)
        img_paths = [path.join(img_dir, img_file) for img_file in img_files]
        for img_path in img_paths:

            img = preprocessing.image.load_img(img_path).convert('RGB')
            d_width, d_height = EastPreprocess.resize_img(img, max_predict_img_size)
            scale_ratio_w, scale_ratio_h = img.width / d_width, img.height / d_height
            img = img.resize((d_width, d_height), Image.BICUBIC)

            array_img = preprocessing.image.img_to_array(img)
            # 封装多张图片，但在此只封装一张
            array_img_all = np.zeros((num_img, d_height, d_width, 3))
            array_img_all[0] = array_img

            tf_img = preprocess_input(array_img, mode='tf')
            x = np.zeros((num_img, d_height, d_width, 3))
            x[0] = tf_img
            y_pred = self.east_model.predict(x)

            for i in range(num_img):

                recs_xy_list = []
                y = y_pred[i]
                y[:, :, :3] = EastData.sigmoid(y[:, :, :3])
                condition = np.greater_equal(y[:, :, 0], pixel_threshold)
                activation_pixels = np.where(condition)
                recs_score, recs_after_nms = EastData.nms(y, activation_pixels)

                for score, rec in zip(recs_score, recs_after_nms):
                    if np.amin(score) > 0:
                        rec = np.reshape(rec, (4, 2))
                        rec[:, 0] *= scale_ratio_w
                        rec[:, 1] *= scale_ratio_h
                        rec = np.reshape(rec, (8,)).tolist()
                        recs_xy_list.append(rec)
                txt_name = path.basename(img_path)[:-4] + '.txt'
                RecdataIO.write_rec_txt(recs_xy_list, output_txt_dir, txt_name)

                # TODO：展示预测结果
                if show_predict_img:
                    pass



