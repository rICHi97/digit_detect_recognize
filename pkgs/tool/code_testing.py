# -*- coding: utf-8 -*-
"""
Created on 2021-12-07 22:33:50

@author: Li Zhi
本模块用于对测试其他部分代码，封装测试过程
"""
import os.path as path

from PIL import Image

from . import cfg, image_processing, visualization
from .. import detect_recognize
from ..east import east_data
from ..recdata import rec, recdata_correcting, recdata_io, recdata_processing

PCA = recdata_correcting.PCA
ImageProcess = image_processing.ImageProcess
RecDraw = visualization.RecDraw
EndToEnd = detect_recognize.EndToEnd
EastPreprocess = east_data.EastPreprocess
Rec = rec.Rec
Recdata = recdata_processing.Recdata
RecdataProcess = recdata_processing.RecdataProcess
RecdataIO = recdata_io.RecdataIO
RecdataRecognize = recdata_processing.RecdataRecognize
# 顺序和east网络输出不同，是先输出左边一列再右边，这种情况下似乎会导致基于PCA的分组失效
two_col_recs_xy_list = [[253.54, 132.0, 207.73, 135.41, 208.97, 187.1, 255.4, 180.6], [254.16, 181.84, 209.9, 188.03, 211.45, 235.38, 255.71, 229.81], [255.09, 231.98, 211.75, 237.55, 212.68, 284.29, 258.18, 279.96], [256.33, 282.12, 212.37, 285.84, 212.99, 333.2, 257.57, 330.1], [257.91, 331.48, 214.21, 334.52, 215.72, 381.86, 259.12, 379.13], [260.64, 380.95, 215.72, 382.77, 217.54, 429.2, 262.46, 428.29], [262.29, 430.48, 217.56, 430.75, 220.25, 477.91, 264.72, 477.37], [264.72, 479.79, 221.6, 479.26, 223.49, 525.06, 266.87, 524.79], [267.14, 526.68, 223.49, 526.14, 224.57, 572.49, 269.03, 572.49], [269.57, 574.38, 225.11, 573.57, 226.99, 621.8, 271.99, 621.8], [272.53, 628.27, 228.07, 623.42, 229.15, 670.58, 274.42, 675.16], [276.03, 683.51, 231.3, 677.85, 232.38, 723.66, 277.92, 729.32], [275.55, 730.33, 231.98, 725.6, 232.73, 769.41, 277.04, 776.13], [277.29, 778.13, 232.98, 771.4, 235.22, 816.21, 279.78, 823.43], [280.77, 824.68, 235.22, 818.45, 237.21, 862.77, 282.77, 868.99], [282.52, 870.73, 237.71, 863.51, 238.45, 906.83, 283.01, 915.54], [282.52, 918.03, 238.2, 908.57, 238.95, 952.39, 283.76, 962.1], [284.26, 964.09, 240.44, 953.88, 241.19, 996.95, 285.5, 1008.65], [284.39, 1010.11, 240.94, 998.94, 241.78, 1042.81, 285.75, 1054.53], [286.65, 1059.72, 243.58, 1046.86, 244.94, 1090.83, 287.55, 1103.46], [290.12, 1107.78, 247.27, 1094.93, 248.39, 1137.32, 291.92, 1150.18], [293.06, 1152.29, 249.7, 1139.3, 251.05, 1181.82, 295.08, 1194.98], [294.07, 1196.66, 251.22, 1183.84, 252.4, 1225.34, 295.42, 1239.51], [295.92, 1242.05, 252.4, 1227.37, 252.9, 1269.71, 297.44, 1285.24], [297.27, 1287.09, 253.24, 1272.08, 254.76, 1314.42, 298.45, 1329.61], [299.44, 1332.27, 255.43, 1316.22, 256.19, 1358.55, 300.81, 1374.44], [299.22, 1375.53, 256.74, 1360.56, 258.03, 1402.75, 301.09, 1418.3], [301.23, 1420.03, 259.04, 1405.05, 260.63, 1446.09, 302.96, 1462.5], [303.53, 1466.25, 260.19, 1448.25, 261.49, 1489.0, 304.97, 1507.28], [306.27, 1516.36, 263.22, 1495.48, 264.08, 1537.81, 306.13, 1557.97], [306.27, 1561.86, 263.65, 1540.84, 263.79, 1582.45, 307.28, 1603.62], [304.69, 1606.35, 263.22, 1586.91, 264.37, 1628.09, 306.27, 1648.54], [306.41, 1651.13, 264.37, 1631.12, 265.23, 1670.43, 308.0, 1693.04], [306.41, 1695.34, 265.23, 1673.6, 266.67, 1714.06, 308.0, 1735.22], [309.29, 1738.68, 267.68, 1717.08, 268.26, 1757.4, 310.45, 1780.01], [310.01, 1783.17, 268.98, 1759.7, 269.98, 1799.88, 310.73, 1823.2], [311.74, 1825.36, 269.7, 1801.32, 269.84, 1841.35, 311.74, 1865.97], [311.31, 1868.56, 271.28, 1843.51, 271.14, 1884.54, 311.17, 1909.6], [311.17, 1911.33, 271.28, 1886.56, 271.71, 1926.88, 311.02, 1954.09], [313.76, 1958.27, 273.15, 1932.06, 273.58, 1971.23, 314.19, 1999.59], [314.94, 2004.85, 273.39, 1979.7, 274.29, 2019.23, 315.84, 2047.3], [315.84, 2049.55, 274.29, 2021.7, 275.64, 2061.9, 317.86, 2091.54], [316.74, 2092.89, 275.86, 2064.37, 276.76, 2102.77, 317.19, 2131.29], [317.19, 2133.99, 276.76, 2105.02, 276.99, 2143.64, 317.19, 2174.19], [590.83, 2111.01, 527.99, 2138.13, 529.32, 2178.81, 591.82, 2150.37], [591.16, 2068.68, 528.32, 2096.46, 528.32, 2136.15, 591.16, 2109.36], [592.4, 2026.28, 528.24, 2053.47, 528.24, 2093.5, 592.09, 2066.91], [591.38, 1983.69, 529.18, 2009.7, 529.35, 2050.44, 591.81, 2023.69], [588.82, 1939.84, 525.49, 1965.94, 525.28, 2005.3, 589.03, 1981.12], [589.13, 1890.55, 525.43, 1914.24, 525.49, 1955.24, 588.78, 1930.03], [588.08, 1847.02, 525.25, 1870.89, 525.6, 1911.61, 588.96, 1887.39], [588.37, 1804.67, 524.37, 1827.66, 525.19, 1869.16, 588.86, 1844.52], [588.01, 1761.26, 523.34, 1784.32, 523.97, 1825.05, 588.26, 1802.49], [587.01, 1718.27, 522.84, 1739.45, 523.59, 1781.93, 587.64, 1759.5], [587.14, 1676.53, 522.59, 1696.21, 522.97, 1737.07, 586.76, 1716.01], [586.13, 1632.54, 522.09, 1652.47, 522.84, 1694.33, 586.76, 1673.9], [587.01, 1590.06, 522.84, 1609.73, 522.72, 1650.71, 586.26, 1630.16], [586.51, 1545.56, 521.59, 1563.23, 521.84, 1605.6, 586.01, 1586.04], [586.26, 1502.32, 521.21, 1518.24, 521.09, 1560.1, 585.63, 1542.68], [583.5, 1454.82, 519.21, 1471.24, 521.59, 1513.85, 584.76, 1495.18], [583.38, 1408.83, 518.08, 1426.75, 519.21, 1469.24, 583.75, 1450.56], [581.44, 1365.59, 517.1, 1381.82, 517.4, 1424.52, 582.79, 1407.53], [580.69, 1322.3, 516.05, 1337.63, 516.65, 1379.72, 580.99, 1363.49], [579.19, 1277.65, 514.25, 1292.23, 514.4, 1335.38, 580.54, 1319.44], [579.79, 1233.16, 515.15, 1247.59, 515.6, 1289.83, 580.09, 1275.4], [578.58, 1189.56, 513.05, 1202.79, 514.25, 1245.48, 578.74, 1231.35], [577.08, 1144.92, 512.59, 1156.19, 513.65, 1199.79, 579.04, 1187.16], [577.63, 1097.96, 511.31, 1110.87, 511.31, 1154.98, 577.14, 1142.72], [571.75, 1053.36, 505.75, 1062.51, 505.91, 1109.23, 572.4, 1097.47], [573.38, 1005.99, 504.93, 1013.99, 504.93, 1060.23, 572.4, 1049.28], [571.91, 960.25, 506.89, 967.76, 507.22, 1012.03, 573.22, 1004.52], [571.26, 914.9, 503.52, 921.43, 503.81, 966.11, 572.42, 958.42], [571.55, 868.48, 502.21, 874.86, 503.37, 919.54, 571.7, 912.43], [569.6, 823.29, 500.57, 829.53, 501.72, 873.62, 570.75, 867.16], [569.13, 776.2, 499.87, 781.74, 500.34, 827.22, 570.52, 820.99], [567.98, 731.41, 498.72, 735.11, 499.64, 779.89, 570.29, 774.35], [567.29, 683.94, 498.86, 686.17, 499.98, 733.18, 568.13, 729.84], [566.0, 634.16, 499.44, 634.16, 498.39, 682.72, 565.22, 680.89], [563.65, 580.39, 494.74, 582.22, 496.04, 631.29, 564.43, 625.81], [561.84, 532.93, 492.51, 533.19, 493.31, 580.39, 562.91, 577.99], [560.52, 484.69, 492.11, 485.5, 492.65, 532.11, 561.88, 531.29], [558.28, 436.68, 488.75, 435.45, 491.21, 484.06, 561.05, 483.44], [556.91, 388.35, 487.46, 385.97, 488.17, 434.01, 558.58, 435.68], [555.98, 340.97, 485.82, 338.59, 486.4, 384.84, 557.29, 386.86], [554.92, 292.89, 484.78, 289.21, 485.0, 336.18, 555.79, 339.21], [553.84, 243.97, 482.4, 239.2, 483.27, 285.31, 554.49, 290.72], [552.55, 147.06, 482.49, 138.34, 480.74, 187.71, 554.05, 194.94], [553.48, 195.91, 482.06, 188.77, 481.37, 237.14, 553.94, 242.9]]

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
    # @staticmethod
    # def test_joint_rec(
    #         img_path_or_dir=cfg.test_joint_rec_img_path,
    #         recs_xy_list_or_dir=cfg.test_joint_rec_txt_path,
    # ):
    #     img = _get_img(img_path_or_dir)
    #     img_name = path.basename(img_path_or_dir)[:-4]
    #     recs_xy_list = _get_recs_xy_list(recs_xy_list_or_dir)
    #     ImageProcess.joint_rec(img, img_name, recs_xy_list)
    @staticmethod
    def test_preprocess():
        EastPreprocess.preprocess()

    @staticmethod
    def test_label():
        EastPreprocess.label()

    @staticmethod
    def test_pca_divide_groups(
        recs_xy_list=two_col_recs_xy_list,
    ):
        """
        测试对于不同类别的，打乱顺序的（重点）能否分开
        Parameters
        ----------
        Returns
        ----------
        """
        # 1.读取一个recs_xy_list
        # 2.打乱顺序
        # 3.调用pca分组
        reorder_recs_xy_list = RecdataProcess.reorder_recs(recs_xy_list, order='ascending')
        pca_values = PCA.get_pca_values(reorder_recs_xy_list)
        divide_groups = PCA.divide_recs(pca_values)

        return divide_groups

    @staticmethod
    def test_get_text_area(
        img_path=cfg.test_get_text_area_img_path,
        recs_txt_path=cfg.test_get_text_area_txt_path,
    ):
        """
        Parameters
        ----------

        Returns
        ----------
        """
        recs_xy_list, recs_classes_list = RecdataIO.read_rec_txt(recs_txt_path, True)
        recs_text_area = []
        for i, classes in enumerate(recs_classes_list):
            if classes == '编号':
                recs_text_area.append(Recdata.get_text_area(recs_xy_list[i]))
        img = Image.open(img_path)
        RecDraw.draw_recs(recs_text_area, img)
        img.save('text.jpg')

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
        # img = _get_img(img_path)
        img_name = path.basename(img_path)[:-4]
        recs_xy_list, recs_classes_list = RecdataIO.read_rec_txt(
            recs_txt_path, return_classes_list=True
        )
        recs_list = []
        for i in range(len(recs_xy_list)):
            rec = Rec(xy_list=recs_xy_list[i], classes=recs_classes_list[i])
            recs_list.append(rec)
        recdata = RecdataRecognize.recognize(img_name, recs_list)

        return recdata

    @staticmethod
    def test_end_to_end(
        cubicle_id=cfg.test_end_to_end_cubicle_id,
        img_path=cfg.test_end_to_end_img_path,
        test_rec_txt_path=cfg.test_end_to_end_txt_path,
        
    ):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        end_to_end = EndToEnd()
        end_to_end.detect_recognize(img_path)


    @staticmethod
    def test_draw_label_txt(
        img_path=cfg.test_draw_label_txt_img_path,
        label_txt_path=cfg.test_draw_label_txt_label_txt_path,
    ):
        """
        label_txt基于原始大小图片
        Parameters
        ----------

        Returns
        ----------
        """
        img = Image.open(img_path)
        recs_xy_list, recs_classes_list = RecdataIO.read_rec_txt(label_txt_path, True)
        for xy_list, classes in zip(recs_xy_list, recs_classes_list):
            RecDraw.draw_rec(xy_list, img)
            RecDraw.draw_text(classes, xy_list, img)

    @staticmethod
    def test_draw_gt(
        gt_path=cfg.test_draw_gt_gt_path,
        img_path=cfg.test_draw_gt_img_path,
        resized=cfg.test_draw_gt_resized,
    ):
        """
        Parameters
        ----------
        Returns
        ----------
        """
        RecDraw.draw_gt(gt_path, img_path, resized)


class ParamOptimize():
    """
    用于优化代码中的参数
    """
    pass

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
#         visualization.RecDraw.draw_recs_by_txt(label_path, img, 2, 'black', True)
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
#     # visualization.RecDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
#     # visualization.RecDraw.draw_recs(original_recs_shape_data, img, 2, 'black', True)
#     visualization.RecDraw.draw_recs(corrected_recs_shape_data, img, 2, 'black', True)
#     img.save(img_test_name)

# # 矫正多张图片
# if test_correct_all_imgs:
#     imgs_rec_dict = recdata_io.RecdataIO.read_rec_txt_dir('./resource/test_data/image_txt')
#     i = 0
#     imgs_xy_list = {}
#     for key, recs_xy_list in imgs_rec_dict.items():
#         img_name = key[:-4]
#         try:
#             img = Image.open('./resource/test_data/image/' + img_name + '.jpg')
#         except FileNotFoundError:
#             img = Image.open('./resource/test_data/image/' + img_name + '.png')
#         RecDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
#         # if len(recs_xy_list) < 3:
#         #     i += 1
#         # else:
#         #     corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
#         #     _ = []
#         #     for rec_shape_data in corrected_recs_shape_data:
#         #         xy_list = recdata_processing.Recdata.get_xy_list(rec_shape_data)
#         #         visualization.RecDraw.draw_rec(
#         #             xy_list, img, width=2, color='black', distinguish_first_side=True

#         #         )
#         #         _.append(xy_list)
#         #     imgs_rec_dict[key] = _
#         img.save('./resource/test_data/' + img_name + '.jpg')

# if test_east_predict:

#     east = EastNet()
#     east.predict()
#     imgs_rec_dict = RecdataIO.read_rec_txt_dir('./resource/test_data/image_txt')
#     i = 0
#     imgs_xy_list = {}
#     for key, recs_xy_list in imgs_rec_dict.items():
#         img_name = key[:-4]
#         try:
#             img = Image.open('./resource/test_data/image/' + img_name + '.jpg')
#         except FileNotFoundError:
#             img = Image.open('./resource/test_data/image/' + img_name + '.png')
#         RecDraw.draw_recs(recs_xy_list, img, 2, 'black', True)
#         # if len(recs_xy_list) < 3:
#         #     i += 1
#         # else:
#         #     corrected_recs_shape_data = recdata_correcting.Correction.correct_rec(recs_xy_list)
#         #     _ = []
#         #     for rec_shape_data in corrected_recs_shape_data:
#         #         xy_list = recdata_processing.Recdata.get_xy_list(rec_shape_data)
#         #         visualization.RecDraw.draw_rec(
#         #             xy_list, img, width=2, color='black', distinguish_first_side=True

#         #         )
#         #         _.append(xy_list)
#         #     imgs_rec_dict[key] = _
#         img.save('./resource/test_data/' + img_name + '.jpg')

# if test_show_gt:

#     gt_filepath = './resource/train_data/b_train_label/terminal_5_number_1_gt.npy'
#     img_filepath = './resource/train_data/a_img/terminal_5_number_1.jpg'
#     RecDraw.draw_gt_file(gt_filepath, img_filepath)