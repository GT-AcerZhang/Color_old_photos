# Author: Acer Zhang
# Datetime:2020/7/8 21:38
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import numpy as np
import cv2 as cv

BLOCK_SIZE1D = 50  # 1D细分类块大小，越大分类数越高，效果越好 - 模型大小会增加
MAX_STEP = 6  # 最大步长，越低效果越好
MIN_STEP = 3  # 最低步长
IMG_BLOCK = 10000  # 图片采样数，越高越好 - 统计速度变慢
R_IMG_SIZE = (1024, 1024)  # 采样分辨率，越高越好 - 可能出现爆炸现象 - 性价比较低
IMAGE_DIR_PATH = "./test2017"
FILE_NAME = "color_files/Color1D_Large"


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
    block_list = dict()
    color_list = dict()
    while True:
        if (label == BLOCK_SIZE1D - 2 and end - start <= MAX_STEP) or start + MAX_STEP >= 255:
            block_list.update(dict([(k, label) for k in range(start, 256)]))
            color_list[label] = int((np.max(signal[start:256]).astype("uint8") + (end - start)) * 0.5)
            break
        if np.sum(signal[start:end]) >= int(step) or end - start >= MAX_STEP:
            block_list.update(dict([(k, label) for k in range(start, end)]))
            color_list[label] = start + int((np.argmax(signal[start:end]).astype("uint8") + (end - start)) * 0.5)
            start = end
            end += MIN_STEP
            label += 1
        else:
            end += 1
    return block_list, color_list, label + 1


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

a_dict, al_dict, a_num = analysis1d(signal_cache_a, a_step)
b_dict, bl_dict, b_num = analysis1d(signal_cache_b, b_step)
print("写入硬盘...")
with open("./" + FILE_NAME + ".dict", "w", encoding="utf-8") as f:
    f.write(str([[a_dict, b_dict], [al_dict, bl_dict]]))
with open("./" + FILE_NAME + "_" + str(a_num * b_num) + ".info", "w", encoding="utf-8") as f:
    f.write("分类数为:" + str(a_num * b_num) + "\tA:" + str(a_num) + "\tB:" + str(b_num) + "\nBLOCK_SIZE1D:" +
            str(BLOCK_SIZE1D) + "\tMAX_STEP:" + str(MAX_STEP) + "\tMIN_STEP:" + str(MIN_STEP))
    print("已保存至./Color1D.dict \n分类数为:", a_num * b_num, "\tA:", a_num, "\tB:", b_num)
