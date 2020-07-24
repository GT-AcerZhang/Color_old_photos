import os
from multiprocessing import Pool

import numpy as np
import cv2 as cv

POOL_NUM = 4
DICT_FILE_PATH = "./Color1D.dict"
DATA_PATH = "data/f"
OUT_PATH = "./data"
IMG_RESIZE = (256, 256)
TEST_MOD = 5


def cvt_color(ori_img, color_dict: dict):
    h, w = ori_img.shape
    for pix_h in range(h):
        for pix_w in range(w):
            ori_img[pix_h][pix_w] = color_dict[ori_img[pix_h][pix_w]]
    return ori_img


def cvt_process(file_name, img_id, img_size, color_dict, data_p, out_p):
    file_path = os.path.join(data_p, file_name)
    a_dict, b_dict = color_dict
    img = cv.imread(file_path)
    r_img = cv.resize(img, (img_size[0] * 2, img_size[1] * 2))
    o_l, _, _ = cv.split(r_img)
    if np.sum(o_l) > np.sum(_):
        return str(img_id) + "L"
    img = cv.resize(img, img_size)
    im = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(im)
    label_a = cvt_color(a, a_dict)
    label_b = cvt_color(b, b_dict)
    r_im = cv.merge([l, label_a, label_b])
    np.save(os.path.join(out_p, "lab", str(img_id)), r_im)
    np.save(os.path.join(out_p, "label", str(img_id)), o_l)
    return img_id


def print_log(ipt):
    print("POOL ID:", ipt, "\t--OK")


def print_error(ipt):
    print("ERROR!", ipt)


if __name__ == '__main__':
    with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
        c_dict = eval(f.read())[0]
    mk_list = [os.path.join(OUT_PATH, "train"), os.path.join(OUT_PATH, "test"),
               os.path.join(OUT_PATH, "train/lab"), os.path.join(OUT_PATH, "test/lab"),
               os.path.join(OUT_PATH, "train/label"), os.path.join(OUT_PATH, "test/label")]
    for p in mk_list:
        if not os.path.exists(p):
            os.makedirs(p)
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
