# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:03:02 2021

@author: LIZHi
"""
from PIL import  Image

threshold = 1.4
# 按区间数划分长度
# 参数：总长度、区间数、每段区间长度
def length_segment(length, cnt_length_segment, length_each_segment, overlapping_ratio = 0.1, threshold = threshold):
    length_segment_coord_list = []
    # 当图片某边长度不是过长（小于east输入最大尺寸的threshold倍）时就不进行分割
    if cnt_length_segment < threshold:
        length_segment_coord = (0, length)
        length_segment_coord_list.append(length_segment_coord)
        return length_segment_coord_list
    else:
        cnt_length_segment = int(cnt_length_segment + 1)
        overlapping_length = overlapping_ratio * length_each_segment
        for i in range(cnt_length_segment):
            # 分割的区间长度，为了保证不漏检，使相邻两个区间存在重叠部分
            # 先计算初始每个区间的长度，然后对于每个刻度，如果它的另一侧刻度固定（即最边上的两个端点），则延长百分之20区间宽度，否则延长百分之10
            # 起始区间
            if i == 0:
                right_coord = int(length_each_segment + 2 * overlapping_length + 1)
                length_segment_coord = (0, right_coord)
            # 结尾区间
            elif i == cnt_length_segment - 1 :
                # 由于是数轴上往左，所以不需要额外减1再取int
                left_coord = int(length - length_each_segment - 2 * overlapping_length)
                length_segment_coord = (left_coord, length)
            else:
                left_coord  = int(   i    * length_each_segment - overlapping_length)
                right_coord = int((i + 1) * length_each_segment + overlapping_length + 1)
                length_segment_coord = (left_coord, right_coord)
            length_segment_coord_list.append(length_segment_coord)
        return length_segment_coord_list

def segment_img_to_east(img_path, out_path, max_east_img_size = 832, threshold = threshold):
    img_name = img_path.split('/')[-1][:-4]
    img = Image.open(img_path)
    width, height = img.width, img.height
    # 用于存储分割后的各区间坐标
    width_segment_coord_list, height_segment_coord_list = [], []
    # 分割数
    cnt_width_segment, cnt_height_segment = width / max_east_img_size, height / max_east_img_size  
    width_each_segment, height_each_segment = width / cnt_width_segment, height / cnt_height_segment
    # 分割后的每段区间坐标，用于裁切图片
    width_segment_coord_list  = length_segment(width,  cnt_width_segment,  width_each_segment)
    height_segment_coord_list = length_segment(height, cnt_height_segment, height_each_segment)    
    
    # 小于1.4的设置为1，大于1.4的设置为向上取整
    if cnt_width_segment < threshold:
        cnt_width_segment = 1
    else:
        cnt_width_segment = int(cnt_width_segment + 1)
    
    if cnt_height_segment < threshold:
        cnt_height_segment = 1
    else:
        cnt_height_segment = int(cnt_height_segment + 1)

    for i in range(cnt_width_segment):
        for j in range(cnt_height_segment):
            width_segment_coord  = width_segment_coord_list[i]
            height_segment_coord = height_segment_coord_list[j]
            crop_region = (width_segment_coord[0], height_segment_coord[0], 
                           width_segment_coord[1], height_segment_coord[1])
            img_segment = img.crop(crop_region)
            img_segment.save(out_path + r'\%s_W%dH%d.jpg'%(img_name, i, j))
            
        