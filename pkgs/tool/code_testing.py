# -*- coding: utf-8 -*-
"""
Created on 2021-12-07 22:33:50

@author: Li Zhi
"""
import os.path as path

from PIL import Image

from . import cfg, image_processing, visualization
from ..recdata import recdata_io, recdata_processing

ImageProcess = image_processing.ImageProcess
ImageDraw = visualization.ImageDraw
RecdataIO = recdata_io.RecdataIO
RecdataRecognize = recdata_processing.RecdataRecognize

# TODO：路径检查
def _get_img(img):

    if isinstance(img, Image.Image):
        pass
    elif isinstance(img, str):
        img = Image.open(img)

    return img

def _get_recs_xy_list(recs_xy_list):
    if isinstance(recs_xy_list, list):
        pass
    elif isinstance(recs_xy_list, str):
        recs_xy_list = RecdataIO.read_rec_txt(recs_xy_list)

    return recs_xy_list


class CodeTest(object):

    # TODO：测试对整个文件夹进行操作，目前仅支持操作单张img及recs_xy_list
    # TODO：参数的命名可能要考虑
    @staticmethod
    def test_joint_rec(
        img_path_or_dir=cfg.test_joint_rec_img_path, 
        recs_xy_list_or_dir=cfg.test_joint_rec_txt_path,
    ):
        img = _get_img(img_path_or_dir)
        img_name = path.basename(img_path_or_dir)[:-4]
        recs_xy_list = _get_recs_xy_list(recs_xy_list_or_dir)
        ImageProcess.joint_rec(img, img_name, recs_xy_list)

    @staticmethod
    def test_recognize(
        img_path=cfg.test_recognize_img_path,
        recs_txt_path=cfg.test_recognize_recs_txt_path,
    ):
        """
        Parameters
        ----------
        img_path：图片路径
        recs_txt_path：label txt，每行数据包括4个端点坐标和类别信息
        Returns
        ----------
        """    
        img = _get_img(img_path)
        img_name = path.basename(img_path)[:-4] 
        recs_xy_list, recs_classes_list = RecdataIO.read_rec_txt(
            recs_txt_path, return_classes_list=True
        )
        RecdataRecognize.recognize(img, img_name, recs_xy_list, recs_classes_list)



# img_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\img').replace('\\', '/')
# label_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\txt_合并').replace('\\', '/')
# output_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\裁切结果').replace('\\', '/')
# json1_dir = path.normpath(r'D:\各种文件\图像识别\端子排数据\标注整个边框\json').replace('\\', '/')
# json2_dir = path.normpath(
#     r'D:\各种文件\图像识别\端子排数据\标注整个边框\json_标注铭牌'
# ).replace('\\', '/')
# # output_dir = path.normpath(
# #     r'D:\各种文件\图像识别\端子排数据\标注整个边框\json_合并'
# # ).replace('\\', '/')

# if test_east_net:

#     east = EastNet()
#     east.east_model.summary()    

# if test_east_data:
        
#     EastPreprocess.preprocess()
#     EastPreprocess.label()

# if test_east_train:

#     callbacks = [
#         EastData.callbacks('early_stopping'),
#         EastData.callbacks('check_point'),
#         EastData.callbacks('reduce_lr'),
#     ]
#     east = EastNet()
#     east.train(callbacks=callbacks)

# if test_label:

#     label_files = os.listdir(label_dir)
#     for file in label_files:
#         label_path = path.join(label_dir, file)
#         img_name = file.replace('.txt', '.jpg')
#         img_path = path.join(img_dir, img_name)
#         if not path.exists(img_path):
#             img_name = file.replace('.txt', '.png')
#             img_path = path.join(img_dir, img_name)
#         img = Image.open(img_path)
#         visualization.ImageDraw.draw_recs_by_txt(label_path, img, 2, 'black', True)
#         img.save(path.join(label_dir, img_name))

# # TODO：检查铭牌标签是否出错
# if test_crop_img:

#     # TODO：注意label
#     image_processing.ImageProcess.random_crop(
#         img_dir, label_dir, output_dir, 50, 0.4, 0.2, 'number'
#     )
#     image_processing.ImageProcess.random_crop(img_dir, label_dir, output_dir, 5, 0.4, 0.2, 'plate')

# if test_merge_json:

#     recdata_io.RecdataIO.merge_json(json1_dir, json2_dir, output_dir, json2_keyword='plate')
#     recdata_io.RecdataIO.json_to_txt(output_dir)

# if test_correct_one_img:
#     txt_name, img_name = '2_original.txt', '2.png'
#     img_test_name = '2_test.jpg'
#     recs_xy_list = recdata_io.RecdataIO.read_rec_txt(txt_name)
#     original_recs_shape_data = []
#     for xy_list in recs_xy_list:
#         rec_shape_data = recdata_processing.Recdata.get_rec_shape_data(xy_list)
#         original_recs_shape_data.append(rec_shape_data)
#     img = Image.open(img_name).copy()
#     corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
#     # visualization.ImageDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
#     # visualization.ImageDraw.draw_recs(original_recs_shape_data, img, 2, 'black', True)
#     visualization.ImageDraw.draw_recs(corrected_recs_shape_data, img, 2, 'black', True)
#     img.save(img_test_name)

# # 矫正多张图片
# if test_correct_all_imgs:
#     imgs_rec_dict = recdata_io.RecdataIO.read_rec_txt_dir('./source/test_data/image_txt')
#     i = 0
#     imgs_xy_list = {}
#     for key, recs_xy_list in imgs_rec_dict.items():
#         img_name = key[:-4]
#         try:
#             img = Image.open('./source/test_data/image/' + img_name + '.jpg')
#         except FileNotFoundError:
#             img = Image.open('./source/test_data/image/' + img_name + '.png')
#         ImageDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
#         # if len(recs_xy_list) < 3:
#         #     i += 1
#         # else:
#         #     corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
#         #     _ = []
#         #     for rec_shape_data in corrected_recs_shape_data:
#         #         xy_list = recdata_processing.Recdata.get_xy_list(rec_shape_data)
#         #         visualization.ImageDraw.draw_rec(
#         #             xy_list, img, width=2, color='black', distinguish_first_side=True
                    
#         #         )
#         #         _.append(xy_list)
#         #     imgs_rec_dict[key] = _
#         img.save('./source/test_data/' + img_name + '.jpg')

# if test_east_predict:

#     east = EastNet()
#     east.predict()
#     imgs_rec_dict = RecdataIO.read_rec_txt_dir('./source/test_data/image_txt')
#     i = 0
#     imgs_xy_list = {}
#     for key, recs_xy_list in imgs_rec_dict.items():
#         img_name = key[:-4]
#         try:
#             img = Image.open('./source/test_data/image/' + img_name + '.jpg')
#         except FileNotFoundError:
#             img = Image.open('./source/test_data/image/' + img_name + '.png')
#         ImageDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
#         # if len(recs_xy_list) < 3:
#         #     i += 1
#         # else:
#         #     corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
#         #     _ = []
#         #     for rec_shape_data in corrected_recs_shape_data:
#         #         xy_list = recdata_processing.Recdata.get_xy_list(rec_shape_data)
#         #         visualization.ImageDraw.draw_rec(
#         #             xy_list, img, width=2, color='black', distinguish_first_side=True
                    
#         #         )
#         #         _.append(xy_list)
#         #     imgs_rec_dict[key] = _
#         img.save('./source/test_data/' + img_name + '.jpg')

# if test_show_gt:
    
#     gt_filepath = './source/train_data/b_train_label/terminal_5_number_1_gt.npy'
#     img_filepath = './source/train_data/a_img/terminal_5_number_1.jpg'
#     ImageDraw.draw_gt_file(gt_filepath, img_filepath)