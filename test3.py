# Author: Acer Zhang
# Datetime: 2020/8/20 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# Author: Acer Zhang
# Datetime: 2020/8/13
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os

from tqdm import trange
import numpy as np
import cv2 as cv

DATA_PATH = "/home/aistudio/infer_result/cyclegan"

RESIZE = 512


# 转换图片
def cvt_img(data_path):
    files = os.listdir(data_path)
    for index in trange(len(files)):
        if os.path.splitext(files[index])[1] not in [".jpg", ".jpeg", ".bmp", "png"] and "fake" in \
                os.path.splitext(files[index])[0]:
            print(files[index], "is not right img file, so skip this item!")
            continue
        lab_img = cv.imread(os.path.join(data_path, files[index]))
        l_img = cv.imread(os.path.join(data_path, files[index].replace("fake", "input")))
        try:
            ori_img = cv.resize(lab_img, (RESIZE, RESIZE))
        except Exception as e:
            print(e)
            continue
        l_img = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        l, a, b = cv.split(lab_img)
        im = cv.merge([l_img, a, b])
        im = cv.cvtColor(im, cv.COLOR_LAB2BGR)
        cv.imwrite(os.path.join(data_path, files[index] + "_res"), im)


cvt_img(DATA_PATH)
