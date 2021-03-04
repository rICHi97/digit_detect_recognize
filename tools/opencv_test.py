# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:08:33 2021

@author: LIZHi
"""
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

img_dir = '../east/test/image/'
result_dir = '../results/cv_results/'

img_name = '1.jpg'

# @param:type:'MEAN', 'GAUSSIAN', 'OTSU', 'CANNY'
def edge_detect(img, out_path, thresh_type, morph_type):
    # bilateral filter去噪
    # blur1 = cv.bilateralFilter(img, 9, 75, 75)
    blur = cv.GaussianBlur(img, (3, 3), 0)
    if thresh_type == 'MEAN':
        edges = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2)
    elif thresh_type == 'GAUSSIAN':
        edges = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)
    elif thresh_type == 'OTSU':
        ret, edges = cv.threshold(blur, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # _, edges = cv.threshold(blur, min(255, int(1.2 * ret)), 255, cv.THRESH_BINARY)
    elif thresh_type == 'CANNY':
        ret, _ = cv.threshold(blur, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Canny算法中已经包括高斯模糊，所以传入图片为原始图片
        edges = cv.Canny(img, int(0.8 * ret), int(1.8 * ret))
    # edges = cv.Canny(img, 100, 200)
    if True:
        # kernel = np.ones((5, 5), np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        if morph_type == 'erosion':
            edges = cv.erode(edges, kernel, iterations = 1)
        elif morph_type == 'dilation':
            edges = cv.dilate(edges, kernel, iterations = 1)
        elif morph_type == 'opening':
            edges = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
        elif morph_type == 'closing':
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    return edges

def draw_lines(img, lines):
    for line_points in lines:
        cv.line(img,(line_points[0][0],line_points[0][1]),(line_points[0][2],line_points[0][3]),
                 (0,255,0),2,8,0)
    cv.imshow("line_img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':   
    
    img = cv.imread(img_dir + img_name)
    img_copy = img.copy()
    
    # 转为HLS颜色空间筛选白色
    img_hls = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGRA2BGR),
                          cv.COLOR_BGR2HLS)
    lower_white = np.array([0, 180, 0])
    upper_white = np.array([255, 255, 255])
    mask = cv.inRange(img_hls, lower_white, upper_white)
    img_with_mask = cv.bitwise_and(img, img, mask = mask)
    img_with_mask_gray = cv.cvtColor(img_with_mask, cv.COLOR_BGR2GRAY)
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = edge_detect(img, result_dir, 'OTSU', 'opening')
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(img_with_mask_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[13]
    M = cv.moments(cnt)
    print(M)
    cv.drawContours(img_copy, [cnt], 0, (0, 255, 0), 2)
    cv.imshow('edges', edges)
    cv.imshow('contours', img_copy)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # plt.imshow(img_contours, 'gray')
    # edge_detect(img, result_dir)
    # lines = cv.HoughLinesP(edges, 1, np.pi/180, 70, minLineLength=80, maxLineGap=10)
    # draw_lines(img, lines)
    
# Numpy index
# b = img[:, :, 0]
# g = img[:, :, 1]
# r = img[:, :, 2]
# img1 = cv.merge((r, g, b))
# plt.subplot(122), plt.imshow(img1)
    
# OpenCV split
# for i in range(1000):
#     b, g, r = cv.split(img)
# split比index差不多慢十倍

# 对图片取色，将取色16进制RGB转为10进制BGR
# 输入格式为取色格式，即#xxxxxx，其中x表示一位16进制字符
# def hexRGB_to_decBGR(str_hexRGB):
#     hexRGB_R = '0x' + str_hexRGB[1:3]
#     hexRGB_G = '0x' + str_hexRGB[3:5]
#     hexRGB_B = '0x' + str_hexRGB[5:7]
#     decBGR_B = int(hexRGB_B, 16)
#     decBGR_G = int(hexRGB_G, 16)
#     decBGR_R = int(hexRGB_R, 16)
#     decBGR = [decBGR_B, decBGR_G, decBGR_R]
#     decHLS = cv.cvtColor(np.uint8([[decBGR]]), cv.COLOR_BGR2HLS)
#     return decBGR, decHLS


# Bitwise Operations
# Load two images
# img = cv.imread('test.jpg')
# hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
# lower_white = np.array([0, 120, 0])
# upper_white = np.array([255, 255, 255])
# mask = cv.inRange(hls, lower_white, upper_white)
# img_with_mask = cv.bitwise_and(img, img, mask = mask)
# cv.imshow('original', img)
# cv.imshow('mask', mask)
# cv.imshow('with mask', img_with_mask)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 端子图片取色，统计取色结果



