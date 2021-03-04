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
def joint_img(img_path, out_path, max_width = 1000, img_height = 62, spacing = 20, input_img_height = 40):
    # 字典，存储图片文件名，及该文件名对应图片剩余可用宽度
    # 图片名 = 'joint_img' + '当前rec所属图片' + '_序号'
    joint_img_name_available_width = {} 
    # 字典，存储拼接图片序号
    joint_img_index = {}
    all_imgs = listdir(img_path)
    for img in all_imgs:
        
        rec = Image.open(img_path + r'/%s'%(img))
        rec_width = rec.width
        img_name = img.split('.')[0]
        first_underline = img_name.find('_')
        img_name = img_name[first_underline + 1:]
        prefix = 'joint_img_' + img_name
        
        # 如果前缀不在序号字典中，就创建该前缀并分配初始序号0
        # 如果前缀在序号字典中，组合前缀、rec图片、序号，在文件名及宽度字典中查询可用宽度
        if prefix not in joint_img_index.keys():
            joint_img_index['%s'%(prefix)] = 0
            index = joint_img_index['%s'%(prefix)]
        else:
            index = joint_img_index['%s'%(prefix)]
        joint_img_name = '%s_%d'%(prefix, index)
        
        # 如果图片名不在图片名与宽度字典中，就创建该图片名并分配初始可用宽度
        # 如果在，就计算剩余可用宽度并更新
        # 更新的原则是：如果剩余可用宽度如果小于当前rec宽度，就更新前缀对应序号
        # 并在图片名与宽度字典中创建一个新的图片名，分配一个初始可用宽度
        if joint_img_name not in joint_img_name_available_width.keys():
            joint_img_name_available_width['%s'%(joint_img_name)] = max_width
        elif joint_img_name_available_width['%s'%(joint_img_name)] < rec_width:
            joint_img_index['%s'%(prefix)] += 1
            index = joint_img_index['%s'%(prefix)]
            joint_img_name = '%s_%d'%(prefix, index)
            joint_img_name_available_width['%s'%(joint_img_name)] = max_width
        
        # 根据剩余可用宽度生成一张空白的用于粘贴的图片和粘贴区域
        # 剩余可用宽度为最大宽度时，说明当前需要新建一张图片
        if  joint_img_name_available_width['%s'%(joint_img_name)] == max_width:
            joint_img = Image.new('RGB', (max_width, img_height), 'white')
            paste_position = (0, int((img_height - input_img_height) / 2))
            joint_img.paste(rec, paste_position)
            # 更新可用宽度
            joint_img_name_available_width['%s'%(joint_img_name)] -= rec_width + spacing
        # 说明已经存在一张可用于粘贴的图片
        else:
            joint_img = Image.open(out_path + r'/%s.jpg'%(joint_img_name))
            paste_position = (max_width - joint_img_name_available_width['%s'%(joint_img_name)] - rec_width + spacing, 
                              int((img_height - input_img_height) / 2))
            joint_img.paste(rec, paste_position)
            joint_img_name_available_width['%s'%(joint_img_name)] -= rec_width + spacing
        joint_img.save(out_path + '/%s.jpg'%(joint_img_name))