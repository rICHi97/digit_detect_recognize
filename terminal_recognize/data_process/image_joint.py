# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:05:36 2021

@author: rICHi0923
"""

from os import listdir
from PIL import Image
'''
拼接图片
api接口有次数和qps限制，考虑到各个数字图片较小，将同一张图片的裁剪图片横向拼接在一起
有宽度限制1024，代码中限制到1000，省略了高度一致检查
默认拼接文件名前缀一致的图片
'''
def joint_img_(img_path, out_path, max_width = 1000, img_height = 42, spacing = 20, input_img_height = 30):
    # 字典，存储图片文件名，及该文件名对应图片剩余可用宽度
    # 图片名 = 'joint_img_' + 序号
    group_prefix = img_path.split('/')[-2]
    joint_img_name_available_width = {}
    # 字典，存储每张joint图片所对应的digit图片
    joint_img_name_corresponding_digit = {}
    # 字典，存储每张joint图片对应的每张digit图片的位置
    joint_img_name_corresponding_digit_position = {} 
    all_imgs = listdir(img_path)
    
    def take_index(elem):
        return int(elem.split('.')[0][5:])
    
    all_imgs.sort(key = take_index)
    prefix, index = 'joint_img_', 0
    # 初始joint_img图片名
    joint_img_name = prefix + '%d'%(index)
    joint_img_name_available_width[joint_img_name] = max_width
    joint_img_name_corresponding_digit[joint_img_name] = []
    joint_img_name_corresponding_digit_position[joint_img_name] = []

    digit_start, digit_end = 0, 0
    for img in all_imgs:

        img_name = img.split('.')[0]
        rec = Image.open(img_path + '/' + img)
        rec_width = rec.width
  
        # 更新的原则是：如果剩余可用宽度如果小于当前rec宽度，就更新前缀对应序号
        # 在图片名与宽度字典中创建一个新的图片名，分配一个初始可用宽度；
        # 在图片名与对应digit字典中创建一个新的图片名，分配一个初始空列表

        # 当前图片名可用宽度小于rec_width，更新前缀创建一个新的图片名
        if joint_img_name_available_width[joint_img_name] < rec_width:
            # 更新前缀
            index += 1
            joint_img_name = prefix + '%d'%(index)
            joint_img_name_available_width[joint_img_name] = max_width
            joint_img_name_corresponding_digit[joint_img_name] = []
            joint_img_name_corresponding_digit_position[joint_img_name] = []
            digit_start, digit_end = 0, 0    

        joint_img_name_available_width[joint_img_name] -= rec_width + spacing
        joint_img_name_corresponding_digit[joint_img_name].append(img)

        digit_end = digit_start + rec_width
        joint_img_name_corresponding_digit_position[joint_img_name].append([digit_start, digit_end])
        digit_start = digit_end + spacing

    for key in joint_img_name_corresponding_digit.keys():
        joint_img = Image.new('RGB', (max_width, img_height), 'white')
        corresponding_digit = joint_img_name_corresponding_digit[key]
        paste_x, paste_y = 0, int((img_height - input_img_height) / 2)
        for digit in corresponding_digit:
            rec = Image.open(img_path + '/' + digit)
            rec_width = rec.width
            paste_position = [paste_x, paste_y]
            joint_img.paste(rec, paste_position)
            paste_x += rec_width + spacing
        joint_img.save(out_path + '%s.jpg'%(key))
    # return joint_img_name_corresponding_digit_position
    return joint_img_name_corresponding_digit_position
