# !/usr/bin/python
# -*- coding: utf-8 -*-
"""predict text from images docstrings.

OCR模型(east+crnn)识别图片中的文字, Input:images, Output:text dictionary

    $python predict.py

Version: 0.1
"""

import os
from tensorflow import Session, logging, ConfigProto
from east.net.network import East
from east.predict_east import predict_quad
from crnn.net.network import crnn_network
from crnn.predict_crnn import predict_text
from keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
from data_process.clustering_regression_recs_correction import recs_correction, number_results_correction, LOF, cluster
from east.data.preprocess import resize_image
if True:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0,1"
    # 只显示error
    os.environ['TF_MIN_CPP_LOG_LEVEL']='2'
    logging.set_verbosity(logging.ERROR)
    config=ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.per_process_gpu_memory_fraction=0.8
    config.gpu_options.allow_growth=True
    session=Session(config=config)

# east_model_weights_file = "./east/model/east_model_weights.h5"
east_model_weights_file = "./east/model/weights_3T832.031-0.037.h5"
crnn_model_weights_file = "./crnn/model/crnn_weights-36.hdf5"

root_image = "./east/test/image/"
root_predict = './east/test/predict/'
predict_image_path = './east/test/predict/'

digit_font_path = r'D:\各种文件\图像识别\Data\train_data\fonts/simhei.ttf'
results = './results/'
if __name__ == '__main__':

    # todo east model predict
    east = East()
    east_model = east.east_network()
    east_model.load_weights(east_model_weights_file)
    # east_model.summary()

    # todo crnn model predict
    if True:
        model, crnn_model = crnn_network()
        crnn_model.load_weights(crnn_model_weights_file)
        
    for files in os.listdir(root_image):
        # root_image 等于east/test/image
        img_path = os.path.join(root_image, files)
        im_name = img_path.split('/')[-1][:-4]
        print("path : %s" % img_path)
        print("name : %s" % im_name)

        # fixme height 过长压缩导致无法看清字体
        # fixme 图像h/w比例大于阈值采用裁剪方式识别
        img = image.load_img(img_path).convert('RGB')
        height = img.height
        width = img.width
        scale = height / width
        correction_flag = False
        
        if scale > 1.5 and height > 2560:
            # todo 重叠部分系数(coefficient) = width/10
            coe = 0.1
            height_s = width * (1 - coe)
            for i in range(int(height / height_s + 1)):
                height_y = i * height_s
                pt1 = (0, min(height_y, height - width))
                pt3 = (width , min(height_y + width, height))
                img_crop = img.crop((pt1[0], pt1[1], pt3[0], pt3[1]))

                im_crop_name = str(im_name) + '_%d' % i
                text_recs_all, text_recs_len, img_all = predict_quad(east_model, img_crop, img_name=im_crop_name)
                
                if correction_flag:
                    # recs_after_LOF = LOF(text_recs_all)
                    # slope = DBSCAN_Clustering(recs_after_LOF)
                    recs_corrected = recs_correction(text_recs_all)
                    
                    recs_len_corrected = []
                    recs_len_corrected.append(len(recs_corrected))
                
                if correction_flag:
                    temp_len = len(recs_corrected) - 1
                else:
                    temp_len = len(text_recs_all)
                    
                if temp_len > 0:
                    if correction_flag:
                        texts, lines = predict_text(crnn_model, recs_corrected, recs_len_corrected, img_all, img_name=im_crop_name)
                    else:
                        texts, lines = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_crop_name)
                    # for s in range(len(texts)):
                    #    print("result ：%s" % texts[s])

                    # print("result ：%s" % texts_str)
        else:
            text_recs_all, text_recs_len, img_all = predict_quad(east_model, img, img_name=im_name)
            
            if correction_flag:
                LOF_data = LOF(text_recs_all)
                # slope = DBSCAN_Clustering(recs_after_LOF)
                recs_corrected = recs_correction(text_recs_all)
                 
                recs_len_corrected = []
                recs_len_corrected.append(len(recs_corrected)) 
                temp_len = len(recs_corrected) - 1
            else:
                temp_len = len(text_recs_all)
                
            if temp_len > 0:
                if correction_flag:
                    texts, lines = predict_text(crnn_model, recs_corrected, recs_len_corrected, img_all, img_name=im_name)
                else:
                    texts, lines = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_name)
                # for s in range(len(texts)):
                #    print("result ：%s" % texts[s])

                # print("result ：%s" % texts_str) 
        predict_img = img.copy()
        # 需要修改, 新size
        new_width, new_height = resize_image(predict_img, 832)
        predict_img = predict_img.resize((new_width, new_height), Image.BICUBIC).convert('RGB')
        predict_draw = ImageDraw.Draw(predict_img)
        
        if correction_flag:
            temp_recs = recs_corrected
        else:
            temp_recs = text_recs_all
        for rec in temp_recs:
            predict_draw.line([tuple([rec[0], rec[1]]),
                                tuple([rec[2], rec[3]]),
                                tuple([rec[4], rec[5]]),
                                tuple([rec[6], rec[7]]),
                                tuple([rec[0], rec[1]])], width=1, fill='blue')
        predict_img.save(root_predict + im_name + '_11_.jpg')
        predict_img_name = im_name + '_11_.jpg'    
        results_img = Image.open(predict_image_path + predict_img_name).copy()
        results_draw = ImageDraw.Draw(results_img)
        # numbers_corrected = number_results_correction(texts)
        for i in range(len(lines)):
            digit_font = ImageFont.truetype(digit_font_path, 20)
            line = lines[i].strip().split(',')
            width = results_img.size[0]
            lt_pt = (line[0], line[1])
            rt_pt = (line[2],line[3])
            # TODO:需要考虑文本框和图片边界间是否有足够距离
            # TODO:需要考虑是在左边显示还是右边显示
            # digit = numbers_corrected[i]
            digit = texts[i]    
            digit_width, digit_height = results_draw.textsize(digit, digit_font)
            # results_draw.text((float(lt_pt[0]) - digit_width, float(lt_pt[1])), text = digit, font = digit_font, fill = 'red')
            #  LOF_data = LOF(text_recs_all)
            # cluster_data = cluster(text_recs_all)
            # results_draw.text((float(lt_pt[0]) - digit_width, float(lt_pt[1])), text = '第%d类'%(cluster_data[i]), font = digit_font, fill = 'red')
            # if abs(LOF_data[i]) > 1.2:
            #    results_draw.text((float(lt_pt[0]) - digit_width, float(lt_pt[1])), text = '离群', font = digit_font, fill = 'red')
            
        results_img.save(results + '%s.jpg'%(im_name))
        

            
