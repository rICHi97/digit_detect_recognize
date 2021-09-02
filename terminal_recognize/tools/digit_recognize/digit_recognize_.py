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
def digit_recognize(img_path):    
    string_position = {}
    spacing_threshold = 20
    
    # if response:
    #     print(response.json())
    
    with requests.Session() as s:
        # 只读二进制
        with open(img_path, 'rb') as f:
            img = base64.b64encode(f.read())
        params = {'image':img, 'recognize_granularity':'small'}
        access_token = _get_access_token()
        request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/numbers' + '?access_token=' + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = s.post(request_url, data = params, headers = headers)
        # print(response.json())
        if response.json()['words_result_num'] != 0:
            words_result_num = response.json()['words_result_num']
            words_result = response.json()['words_result']
            for i in range(words_result_num):
                all_chars = words_result[i]['chars']
                temp_string = all_chars[0]['char']
                temp_left, temp_right = all_chars[0]['location']['left'], all_chars[0]['location']['left'] + all_chars[0]['location']['width']
                temp_position = [temp_left, temp_right]
                # 判断该字符和下一字符的间距，如果小于阈值就认定其为一个字符串中的字符
                # 采用字典存储每个编号字符串对应位置
                # 可能存在重复的编号，字符串的最后一位设置为序号，有重复就递增
                index = 0
                for j in range(len(all_chars) - 1):
                    if all_chars[j + 1]['location']['left'] - all_chars[j]['location']['left'] < spacing_threshold:
                        temp_string += all_chars[j + 1]['char']
                        temp_right = all_chars[j + 1]['location']['left'] + all_chars[j + 1]['location']['width']
                    else:
                        temp_right = all_chars[j]['location']['left'] + all_chars[j]['location']['width']
                        temp_position = [temp_left, temp_right]
                        while temp_string + str(index) in string_position.keys():
                            index += 1
                        temp_string += str(index)
                        index = 0 
                        string_position[temp_string] = temp_position
                        temp_string = all_chars[j + 1]['char']
                        temp_left = all_chars[j + 1]['location']['left']
                
                while temp_string + str(index) in string_position.keys():
                    index += 1
                temp_string += str(index) 
                index = 0
                string_position[temp_string] = temp_position
            return string_position

