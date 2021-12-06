# -*- coding: utf-8 -*-
"""
Created on 2021-11-21 02:32:36

@author: Li Zhi
本模块用以实现所检测到的端子的识别
"""
import base64
import json
import os
import requests

from . import cfg

_api_key = '7j3KnKhBfvL5M46GwGIIOCBB'
_secret_key = 'OLjSdoILVVRMiKza088n4RFpWZXd5OKK'
_digit_request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/numbers'
_character_request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic'
_host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={_api_key}&client_secret={_secret_key}'
# TODO：token会变化吗？
_access_token = None


class RecdataRecognize(object):

    @staticmethod
    def _get_access_token():
        global _access_token, _host  #pylint: disable=W0603
        if _access_token is None:
            response = requests.get(_host)
            _access_token = response.json()['access_token']

        return _access_token

    @staticmethod
    def _request_post():
        ...

    @staticmethod
    def character_recognize(img_path):
        with open(img_path, 'rb') as f:
            img = base64.b64encode(f.read())
            params = {'image': img}
            access_token = RecdataRecognize._get_access_token()
            request_url = f'{_character_request_url}?access_token={access_token}'
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            response = requests.post(request_url, data=params, headers=headers)
            if response:
                print(response.json())


    def recognize(img, recs_xy_list, recs_classes_list):

        recs_classes_set = set(recs_classes_list)
        for classes in recs_classes_set:
            recs_same_classes = []
            for i in range(len(recs_classes_list)):
                if recs_classes_list[i] == classes:
                    recs_same_classes.append(recs_xy_list[i])
            if classes == '编号':
                # 矫正，裁切拼接图片，调用数字识别
            elif classes == '铭牌':
                # 单独裁切每个rec，调用文字识别