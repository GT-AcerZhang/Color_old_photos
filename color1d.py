# Author: Acer Zhang
# Datetime:2020/7/8 21:38
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
from copy import deepcopy

import numpy as np
import cv2 as cv

BLOCK_SIZE1D = 100  # 1D细分类块大小，越大分类数越高，效果越好 - 模型大小会增加
MAX_STEP = 7  # 最大步长，越低效果越好
MIN_STEP = 3  # 最低步长
IMG_BLOCK = 5000  # 图片采样数，越高越好 - 统计速度变慢
R_IMG_SIZE = (512, 512)  # 采样分辨率，越高越好 - 可能出现爆炸现象 - 性价比较低
IMAGE_DIR_PATH = "./test2017"
FILE_NAME = "color_files/Color1D_Base"


def analysis1d(signal, step):
    """
    颜色细分类
    :param signal: A或B通道信号
    :param step: 步长
    :return: pix -> label 和 label -> pix 以及 label数量
    """
    start = 0
    end = 1
    label = 0
    label_map = dict()
    color_map = dict()
    w_list = list()
    while True:
        if end >= 255 - MAX_STEP:
            label_map.update(dict([(k, label) for k in range(start, 256)]))
            color_map[label] = start + int((np.argmax(signal[start:256]).astype("uint8") + (end - start) * 0.15))
            w = np.sum(signal[start:256])
            w_list.append(w)
            break
        elif np.sum(signal[start:end]) >= int(step) and end - start >= MAX_STEP:
            label_map.update(dict([(k, label) for k in range(start, end)]))
            color_map[label] = start + int((np.argmax(signal[start:end]).astype("uint8") + (end - start) * 0.15))
            start = end
            end += MIN_STEP
            label += 1
            w = np.sum(signal[start:end])
            w_list.append(w)
        else:
            end += 1
    w_list = np.array(w_list).astype("float32")
    w_sum = np.sum(w_list)
    acc = np.max(w_list) / w_sum
    w_list_t = 1 - (w_list / float(w_sum))
    return label_map, color_map, label + 1, w_list_t.tolist(), acc


def cvt2label(ab_img, label_map):
    result = deepcopy(ab_img).astype("uint8")
    for value in range(256):
        if value in ab_img:
            result[result == value] = label_map[value]
    return result


cvt2color = cvt2label
# def cvt2color(label_img, color_map):
#     # h, w = label_img.shape[:2]
#     # label_img = label_img.astype("uint8")
#     # result = np.zeros([h, w], dtype="uint8")
#     # for pix_h in range(h):
#     #     for pix_w in range(w):
#     #         pix = label_img[pix_h][pix_w]
#     #         result[pix_h][pix_w] = color_map[pix]
#     result = deepcopy(label_img).astype("uint8")
#     for value in range(256):
#         if value in label_img:
#             result[result == value] = color_map[value]
#     return result


if __name__ == '__main__':
    print("开始统计颜色信号...")
    signal_cache_a = np.zeros(256)
    signal_cache_b = np.zeros(256)
    img_num = 0
    for file_id, file in enumerate(os.listdir(IMAGE_DIR_PATH)):
        if os.path.splitext(file)[1] not in [".jpg", ".jpeg", ".bmp"]:
            continue
        img = cv.imread(os.path.join(IMAGE_DIR_PATH, file))
        img = cv.resize(img, R_IMG_SIZE)
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        im_l, im_a, im_b = cv.split(img)
        a = np.array(im_a).flatten()
        b = np.array(im_b).flatten()
        h_a = np.histogram(a, 256, range=(0, 256))[0].astype("float32")
        signal_cache_a += h_a / 1000000
        h_b = np.histogram(b, 256, range=(0, 256))[0].astype("float32")
        signal_cache_b += h_b / 1000000
        img_num += 1
        if img_num == IMG_BLOCK:
            break
        if file_id % 1000 == 0:
            print(file_id, " --OK")
    img_num += 1
    signal_cache_a = signal_cache_a / img_num * 1000000
    signal_cache_b = signal_cache_b / img_num * 1000000

    a_step = np.sum(signal_cache_a) / BLOCK_SIZE1D
    b_step = np.sum(signal_cache_b) / BLOCK_SIZE1D

    print("正在生成字典...")

    a_label_map, a_color_map, a_num, a_w, a_acc = analysis1d(signal_cache_a, a_step)
    b_label_map, b_color_map, b_num, b_w, b_acc = analysis1d(signal_cache_b, b_step)
    print("写入硬盘...")
    with open("./" + FILE_NAME + ".dict", "w", encoding="utf-8") as f:
        f.write(str({"2label": [a_label_map, b_label_map], "2color": [a_color_map, b_color_map], "weight": [a_w, b_w]}))
    with open("./" + FILE_NAME + "_" + str(a_num * b_num) + ".info", "w", encoding="utf-8") as f:
        f.write("分类数为:" + str(a_num * b_num) + "\t(A)" + str(a_num) + "\t(B)" + str(b_num) + "\nBLOCK_SIZE1D:" +
                str(BLOCK_SIZE1D) + "\tMAX_STEP:" + str(MAX_STEP) + "\tMIN_STEP:" + str(MIN_STEP) +
                "\nMIN_ACC:(A)" + str(a_acc) + "\t(B)" + str(b_acc))
        print("已保存至./Color1D.dict \n分类数为:", a_num * b_num, "\tA:", a_num, "\tB:", b_num)
        print("混乱度：", a_acc * b_acc * 100 * 100, "\tA/B:", a_acc, b_acc)

    im = cv.imread("data/train/000000000275.jpg")
    im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
    vdl_l, vdl_a, vdl_b = cv.split(im)
    vdl_a = cvt2label(vdl_a, a_label_map)
    vdl_b = cvt2label(vdl_b, b_label_map)
    vdl_a = cvt2color(vdl_a, a_color_map)
    vdl_b = cvt2color(vdl_b, b_color_map)
    img = cv.merge([vdl_l, vdl_a, vdl_b])
    tmp = cv.cvtColor(img, cv.COLOR_LAB2BGR)
    im = cv.cvtColor(im, cv.COLOR_LAB2BGR)
    cv.imshow("test", tmp)
    cv.imshow("ori", im)
    cv.waitKey(0)
    pass
