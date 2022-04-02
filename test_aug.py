# -*- coding: utf-8 -*-
"""
Created on 2022-04-01 23:18:31

@author: Li Zhi
"""
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import random

def test_aug(img_path):
    ia.seed(random.randint(0, 19970923))
    img = Image.open(img_path)
    img_array = np.asarray(img, dtype=np.uint8)
    seq = iaa.Sequential([
        iaa.SomeOf(
            (0, 3),
            [
                # 随机高斯模糊、均值模糊、中值模糊
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # 随机锐化
                iaa.Sharpen(alpha=(0, 1), lightness=(0.75, 1.25)),

                # 高斯噪声
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=False),

                # 反色
                # iaa.Invert(0.03, per_channel=False),

                # 改变亮度
                iaa.Multiply((0.75, 1.25), per_channel=False),

                # 改变对比度
                iaa.LinearContrast((0.75, 1.25), per_channel=False),
            ],
            random_order=True
        )
    ])

    img_aug = seq.augment_image(img_array)

    return img_aug, seq

if __name__ == '__main__':
    img_aug, seq = test_aug('./video0_0_5_000039.jpg')
    img_aug = Image.fromarray(img_aug)
    img_aug.show()
