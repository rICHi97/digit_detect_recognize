import numpy as np
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm

# 运行入口文件
from . import cfg
from .label import shrink
from math import log
# import cfg
# from label import shrink

# 得到各边向量
# 返回顺序为W1，W2， H1，H2
# 从起始边逆时针顺序
def _get_side_vector(rec):
    return ((rec[0] - rec[2], rec[1] - rec[3]), 
            (rec[6] - rec[4], rec[7] - rec[5]), 
            (rec[2] - rec[4], rec[3] - rec[5]), 
            (rec[0] - rec[6], rec[1] - rec[7]))

# 得到rec中心坐标
def _get_rec_center(rec):
    center_x, center_y = 0, 0
    center_coord = []
    for i in range(4):
        center_x += rec[2 * i] / 4
        center_y += rec[2 * i + 1] / 4
    center_coord = [center_x, center_y]
    return center_x, center_y, center_coord

# 重排rec顺序
# 原始rec四点顺序都是按逆时针给出，但是起点可能错误
# 每个rec为4点坐标，对应四条边。找出两组对边，比较两组对边中最长边
# 选择最长边所在对边组，长边中心y较小的在上方，以此边为开始的一条边重排rec四点顺序
def reorder_rec(rec):
    # 认定W为长边， H为短边
    W1, W2, H1, H2 = _get_side_vector(rec)
    center_W1, center_W2, center_H1, center_H2 = ((rec[0] + rec[2]) / 2, (rec[1] + rec[3]) / 2), \
                                                 ((rec[6] + rec[4]) / 2, (rec[7] + rec[5]) / 2), \
                                                 ((rec[2] + rec[4]) / 2, (rec[3] + rec[5]) / 2), \
                                                 ((rec[0] + rec[6]) / 2, (rec[1] + rec[7]) / 2)
    four_sides = [W1, W2, H1, H2]
    length_sides = []
    for j in range(4):
        length_sides.append(np.linalg.norm(four_sides[j]))
    # 说明W1，W2为最长边所在的一组对边
    if max(length_sides[0], length_sides[1]) >= max(length_sides[2], length_sides[3]):
        # 选择中心y在上方的作为起始边重排顺序
        if center_W1[1] < center_W2[1]:
            reordered_rec = [rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7]]
            return reordered_rec
        else:
            reordered_rec = [rec[4], rec[5], rec[6], rec[7], rec[0], rec[1], rec[2], rec[3]]
            return reordered_rec
    else:
        if center_H1[1] < center_H2[1]:
            reordered_rec = [rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[0], rec[1]]
            return reordered_rec
        else:
            reordered_rec = [rec[6], rec[7], rec[0], rec[1], rec[2], rec[3], rec[4], rec[5]]
            return reordered_rec


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array

# xy_list格式为四行两列
def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
            / (xy_list[index, 0] - xy_list[first_v, 0] + 1e-4)                
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    # 1e-4代替epsilon
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + 1e-4)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    
    # 转为重排后点属性
    # rec_after_reorder = np.reshape(reorder_rec(np.reshape(reorder_xy_list, (1,8))[0]), (4, 2))
    return reorder_xy_list
    # return rec_after_reorder


def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:     # 起到and的作用
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height

    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:                                     
        o_width = im_width
    
    
    # fixme 最多裁剪31个pixel 是否影响边缘效果
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def preprocess():
    # 路径示意：
    # data_dir:
    #     origin_image_dir;
    #     origin_txt_dir;
    #     train_image_dir;
    #     train_label_dir;
    #     show_gt_image_dir_name;
    #     show_act_image_dir_name;
    data_dir = cfg.data_dir
    # origin dir为数据集原始目录 包括image/txt两部分
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    # 不存在train和label文件夹就创建
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)

    # True
    draw_gt_quad = cfg.draw_gt_quad

    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    # 设置训练集
    train_val_set = []
    # tqdm是进度条
    # o_img_fname为数据集中图片名
    
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))): 
        # with open(os.path.join(origin_txt_dir,
        #                    o_img_fname[:-4] + '.txt'), 'r', encoding="UTF-8") as f:
        #     # print("img file: ", o_img_fname[:-4])
        #     # 单张图片标注信息
        #     anno_list = f.readlines()

        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:  
            # d_wight, d_height = resize_image(im)
            # 增加随机resize
            # print(o_img_fname)
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            # Image.NEARSET -> Image.ANTIALIAS
            im = im.resize((d_wight, d_height), Image.ANTIALIAS).convert('RGB')
            show_gt_im = im.copy()

            # draw on the img
            draw = ImageDraw.Draw(show_gt_im)
            with open(os.path.join(origin_txt_dir,
                                   o_img_fname[:-4] + '.txt'), 'r', encoding="UTF-8") as f:
                # print("img file: ", o_img_fname[:-4])
                # 单张图片标注信息
                anno_list = f.readlines()  

            xy_list_array = np.zeros((len(anno_list), 4, 2))
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)

                # xy_list为前面8个数据
                # 转换为4*2二维数组
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                # xy_list = reorder_vertexes(xy_list)
                # 4*2数组是4个点*2坐标（x,y），按比例缩放
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
        
                # 重排点顺序（貌似是从最左边最上方的点开始）
                # 将重排点顺序该句代码移至按比例缩放之前
                # 注意到缩放后，重排会选择错误的边
                xy_list = reorder_vertexes(xy_list)
                
                diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
                diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
                diff = np.concatenate((diff_1to3, diff_4), axis=0)
                # 四向量长度
                dis = np.sqrt(np.sum(np.square(diff), axis=-1))
                W_H_ratio = (dis[1] + dis[3]) / (dis[0] + dis[2])
                # print(W_H_ratio)
                
                xy_list_array[i] = xy_list
                # long_edge说明长边所在对边，等于0为第一条边和第三条边
                # shrink_ratio = 0.2, shrink_side_ratio = 0.6
                
                
                # 四个向量，分别为V21, V32, V43, V14 
                diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
                diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
                diff = np.concatenate((diff_1to3, diff_4), axis=0)
                # 四向量长度
                dis = np.sqrt(np.sum(np.square(diff), axis=-1))
                W_H_ratio = (dis[1] + dis[3]) / (dis[0] + dis[2])
                if W_H_ratio < 1:
                    W_H_ratio = 2*log(W_H_ratio + 1)
                # print(W_H_ratio)
                #print('***********************************************')
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio * W_H_ratio)

                # 可以将cfg中draw_gt_quad设置为fasle，不展示gt图片
                if draw_gt_quad:
                    # 原始框
                    # draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                    #             tuple(xy_list[2]), tuple(xy_list[3]),
                    #             tuple(xy_list[0])
                    #             ],
                    #           width=1, fill='green')
                    
                    # shrink后框
                    draw.line(
                        [
                            tuple(shrink_xy_list[0]),
                            tuple(shrink_xy_list[1]),
                            tuple(shrink_xy_list[2]),
                            tuple(shrink_xy_list[3]),
                            tuple(shrink_xy_list[0]),
                        ],
                        width=1, 
                        fill='blue',
                    )

                    
                    # 应该是用来判断首尾边界的
                    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    # vs = [[[1, 1, 2, 2, 1], [3, 3, 0, 0, 3]]]
                    for q_th in range(2):
                        draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                    tuple(shrink_1[vs[long_edge][q_th][1]]),
                                    tuple(shrink_1[vs[long_edge][q_th][2]]),
                                    tuple(xy_list[vs[long_edge][q_th][3]]),
                                    tuple(xy_list[vs[long_edge][q_th][4]])],
                                  width=1, fill='yellow')
                
            # im为resize之后的原始图片
            if cfg.gen_origin_img:
                im.save(os.path.join(train_image_dir, o_img_fname))
            
            
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '.npy'),
                xy_list_array)
            
            if draw_gt_quad:
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))
            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_wight,
                                                     d_height))

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    # 0.1作为验证集
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])

if __name__ == '__main__':
    print(os.path.abspath('.../source'))
    preprocess()
