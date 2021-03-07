# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:31:51 2021

@author: LIZHi
"""
import requests
import base64
import json
from os import listdir

# 获取access_token
def _get_access_token():
    API_KEY = '7j3KnKhBfvL5M46GwGIIOCBB'
    SECRET_KEY = 'OLjSdoILVVRMiKza088n4RFpWZXd5OKK'
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/numbers"    
    # 获取access_token
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s'%(API_KEY, SECRET_KEY)
    response = requests.get(host)
    # 返回字典
    access_token = (response.json())['access_token']
    return access_token


# 数字识别
def digit_recognize(digit_rec_file_path):    
    result_dic = {}
    all_string = []
    spacing_threshold = 20
    
    # if response:
    #     print(response.json())
    
    with requests.Session() as s:
        # 只读二进制
        with open(digit_rec_file_path, 'rb') as f:
            img = base64.b64encode(f.read())
        params = {'image':img, 'recognize_granularity':'small'}
        access_token = '%s'%(_get_access_token())
        request_url = request_url + '?access_token=' + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = s.post(request_url, data = params, headers = headers)
        print(response.json())
        if (response.json())['words_result_num'] != 0:
            all_chars = response.json()['words_result'][0]['chars']
            # all_string = []
            temp_string = '%s'%(all_chars[0]['char'])
            # 判断该字符和下一字符的间距，如果小于阈值就认定其为一个字符串中的字符
            for i in range(len(all_chars) - 1):
                if all_chars[i + 1]['location']['left'] - all_chars[i]['location']['left'] < spacing_threshold:
                    temp_string += '%s'%(all_chars[i + 1]['char'])
                else:
                    all_string.append(temp_string)
                    temp_string = '%s'%(all_chars[i + 1]['char'])
            all_string.append(temp_string)
            result_dic['%s'%(digit_rec)] = (response.json())['words_result'][0]['words']
