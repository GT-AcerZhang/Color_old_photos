# Author: Acer Zhang
# Datetime:2020/7/9 9:03
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os
from multiprocessing import Pool

import numpy as np
import cv2 as cv

POOL_NUM = 4
DICT_FILE_PATH = "./Color.dict"
DATA_PATH = "./data/val"
OUT_PATH = "./data"
IMG_RESIZE = (256, 256)
TEST_MOD = 5


def cvt_color(img_a, img_b, color_dict: dict):
    h, w = img_a.shape
    tmp_img = np.zeros((h, w)).astype("int64")
    for pix_h in range(h):
        for pix_w in range(w):
            tmp_img[pix_h][pix_w] = color_dict[img_a[pix_w][pix_h]][img_b[pix_w][pix_h]]
    return tmp_img


def cvt_process(file_name, img_id, img_size, color_dict, data_path, out_path):
    file_path = os.path.join(data_path, file_name)
    img = cv.imread(file_path)
    img = cv.resize(img, img_size)
    im = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(im)
    label = cvt_color(a, b, color_dict)
    im_l = np.expand_dims(l, axis=0)
    im_label = np.expand_dims(label, axis=0)
    final = np.concatenate([im_l, im_label], axis=0)
    np.save(os.path.join(out_path, file_name), final)
    return img_id


def print_log(ipt):
    print("POOL ID:", ipt, "\t--OK")


def print_error(ipt):
    print("ERROR!", ipt)


if __name__ == '__main__':
    with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
        c_dict = eval(f.read())[0]
    mk_list = [os.path.join(OUT_PATH, "train"), os.path.join(OUT_PATH, "test")]
    for p in mk_list:
        if not os.path.exists(p):
            os.makedirs(OUT_PATH)
    print("开始转换...")
    pool_list = Pool(POOL_NUM)
    files_name = os.listdir(DATA_PATH)
    for file_id, file in enumerate(files_name):
        out_path = mk_list[0] if file_id % TEST_MOD != 0 else mk_list[1]
        pool_list.apply_async(func=cvt_process,
                              args=(file, file_id, IMG_RESIZE, c_dict, DATA_PATH, out_path),
                              callback=print_log,
                              error_callback=print_error)
    pool_list.close()
    pool_list.join()
    print("ALL --OK")
