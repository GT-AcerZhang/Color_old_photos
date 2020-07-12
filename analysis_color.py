# Author: Acer Zhang
# Datetime:2020/7/8 21:38
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

CLASS_NUM = 300  # 颜色分类数，越大分类数越高，效果越好 - 模型大小会增加
IMG_BLOCK = 5000  # 图片采样数，越高越好 - 统计速度变慢
R_IMG_SIZE = (512, 512)  # 采样分辨率，越高越好 - 可能出现爆炸现象
IMAGE_DIR_PATH = "val2017"


def cvt_infer(c_map):
    l_map = dict()
    for c in range(CLASS_NUM):
        tmp_a = list()
        for label_a in range(255):
            if c in c_map[label_a]:
                tmp_a.append(label_a)
        if len(tmp_a) > 1:
            label_a = tmp_a[len(tmp_a) // 2]
        elif len(tmp_a) == 1:
            label_a = tmp_a[0]
        else:
            continue
        tmp_b = list()
        for label_b in range(255):
            if c == c_map[label_a][label_b]:
                tmp_b.append(label_b)
        label_b = tmp_b[len(tmp_b) // 2]
        if len(tmp_b) == 1:
            label_b = tmp_b[0]
        l_map[c] = (label_a, label_b)
    return l_map


def k_mean(ipt):
    model = KMeans(n_clusters=CLASS_NUM)
    model = model.fit(ipt)
    model_out = model.predict(ipt)
    k_map = np.zeros((256, 256)).astype("int64")
    for index_x in range(256):
        for index_y in range(256):
            k_map[index_x][index_y] = model_out[index_x * 256 + index_y]
    return k_map


print("开始统计颜色信号...")
signal_cache = np.zeros([256, 256]).astype("float16")
img_num = 0
for file_id, file in enumerate(os.listdir(IMAGE_DIR_PATH)):
    if file_id == IMG_BLOCK:
        break
    img = cv.imread(os.path.join(IMAGE_DIR_PATH, file))
    img = cv.resize(img, R_IMG_SIZE)
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    im_l, im_a, im_b = cv.split(img)
    a = np.array(im_a)
    b = np.array(im_b)
    for pix_x in range(256):
        for pix_y in range(256):
            signal_cache[a[pix_x][pix_y], a[pix_x][pix_y]] += 1e-5
    print(file_id, "--OK!")
signal_max = np.max(signal_cache)
signal_min = np.min(signal_cache)
rate = (signal_max - signal_min) / 256
signal_cache /= rate

signal = [[x, y, signal_cache[x][y]] for x in range(256) for y in range(256)]
print("正在生成字典...")

color_map = k_mean(signal)
label_map = cvt_infer(color_map)
print("写入硬盘...")
with open("./Color.dict", "w", encoding="utf-8")as f:
    f.write(str([color_map.tolist(), label_map]))
print("已保存至./Color.dict \n分类数为:", CLASS_NUM)
print("调用方法为:label = map[A值][B值]")
