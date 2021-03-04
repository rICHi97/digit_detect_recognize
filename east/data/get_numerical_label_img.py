import os
from tqdm import tqdm
# from . import cfg
import cfg
def is_alnum(word):
    try:
        return word.encode('ascii').isalnum()
    except UnicodeEncodeError:
        return False

def get_numerical_label():
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

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))

    # tqdm是进度条
    # o_img_fname为数据集中图片名

    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):

        with open(os.path.join(origin_txt_dir,
                           o_img_fname[:-4] + '.txt'), 'r+', encoding="UTF-8") as f:
            # print("img file: ", o_img_fname[:-4])
            # 单张图片标注信息
            lines = f.readlines()

        with open(os.path.join(origin_txt_dir,
                           o_img_fname[:-4] + '.txt'), 'w', encoding="UTF-8") as f_w:
            # 计数，一张图片有多少个label是数字
            cnt_label_is_number = 0
            # 得到每张图片有多少lebel为数字
            for line in lines:
                tmp_label = line.strip().split(',')
                # if tmp_label[-1].isdigit() or '#' in tmp_label[-1]:
                # if tmp_label[-1].isdigit() or len(tmp_label) == 9:
                if len(tmp_label) == 9:
                # if is_alnum(tmp_label[-1]):
                    cnt_label_is_number += 1
                    f_w.write(line)
                # 如果标签不是数字直接返回
                else:
                    continue
            
            #print("%d个label为数字"%(cnt_label_is_number))

        # 如果计数为0，说明该张图片全部label都不是数字，不作为训练样本
        if cnt_label_is_number == 0:
            # print('该图片无数字或英文字符')
            os.remove(os.path.join(origin_txt_dir,o_img_fname[:-4] + '.txt'))
            os.remove(os.path.join(origin_image_dir,o_img_fname))


if __name__ == '__main__':
    get_numerical_label()
