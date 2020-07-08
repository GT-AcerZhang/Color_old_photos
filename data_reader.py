# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import numpy as np
import cv2 as cv


def reader(data_path, target_size: list, is_val: bool = False):
    file_names = os.listdir(data_path)
    th, tw = target_size

    def _reader():
        for file_name in file_names:
            try:
                ori_img = cv.imread(os.path.join(data_path, file_name))
                ori_l = cv.split(ori_img)[0]
                ori_h, ori_w = ori_img.shape[:-1]
                ori_img = cv.cvtColor(ori_img, cv.COLOR_BGR2LAB)
                ori_img = cv.resize(ori_img, (th, tw))
                l, a, b = cv.split(ori_img)
                im_l = np.array(l).reshape((1, 1, th, tw)).astype("float32")
                im_l /= 255
                if is_val:
                    yield im_l, ori_l, ori_h, ori_w
                else:
                    im_a = np.array(a).reshape((1, 1, th, tw)).astype("float32")
                    im_b = np.array(b).reshape((1, 1, th, tw)).astype("float32")
                    label = np.concatenate([im_a, im_b], axis=1)
                    label = np.array(label)
                    label /= 255
                    yield im_l, label
            except Exception as e:
                print(e)

    return _reader
