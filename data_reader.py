import os

import numpy as np
import cv2 as cv
import PIL.Image as Image  # 随后转为cv2

DICT_FILE_PATH = "./Color1D.dict"
DATA_PATH = "./data/val"


def cvt_color(ori_img, color_dict: dict):
    h, w = ori_img.shape
    for pix_h in range(h):
        for pix_w in range(w):
            ori_img[pix_h][pix_w] = color_dict[ori_img[pix_h][pix_w]]
    return ori_img


def cvt_process(ori_img, color_dict):
    a_dict, b_dict = color_dict
    l, a, b = cv.split(ori_img)
    label_a = cvt_color(a, a_dict)
    label_b = cvt_color(b, b_dict)
    return l, label_a, label_b


def req_weight(im):
    """
    获取权重
    """
    count = np.histogram(im, 256, range=(0, 256))[0].astype("float32")
    count[count == 0] = 1.
    count_t = np.reciprocal(count)
    color_map = dict([(k, v) for k, v in zip(range(256), count_t)])

    im = im.copy().astype("float32")
    for pix_value in range(256):
        if pix_value in im:
            im[im == pix_value] = max(.001, color_map[pix_value])
    return im


def reader(data_path, is_val: bool = False, im_size: list = None):
    file_names = os.listdir(data_path)
    with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
        c_dict = eval(f.read())[0]

    def _reader():
        for file_name in file_names:
            try:
                if is_val:
                    ori_img = Image.open(os.path.join(data_path, file_name)).convert("L")
                    ori_w, ori_h = ori_img.size
                    im_l = ori_img.resize(im_size)
                    im_l = np.array(im_l).reshape((1, 1, im_size[0], im_size[1])).astype("float32")
                    ori_l = np.array(ori_img).reshape((ori_h, ori_w)).astype("uint8")
                    yield im_l / 255, ori_l, ori_h, ori_w
                else:
                    ori_img = cv.imread(os.path.join(data_path, file_name))
                    ori_img = cv.resize(ori_img, (im_size[0] * 2, im_size[1] * 2))
                    ori_l = cv.split(ori_img)[0]
                    re_img = cv.resize(ori_img, (im_size[0], im_size[1]))
                    r_l, r_a, r_b = cvt_process(re_img, c_dict)
                    if np.sum(r_l) < np.sum(r_b):
                        continue
                    ori_l = np.array(ori_l).reshape((1, 1, im_size[0] * 2, im_size[1] * 2)).astype("float32")
                    a_w = req_weight(r_a)
                    b_w = req_weight(r_b)
                    r_l = np.array(r_l).reshape((1, 1, im_size[0], im_size[1])).astype("float32")
                    r_a = np.array(r_a).reshape((1, 1, im_size[0], im_size[1])).astype("int64")
                    r_b = np.array(r_b).reshape((1, 1, im_size[0], im_size[1])).astype("int64")

                    yield r_l / 255, ori_l / 255, r_a, r_b, a_w, b_w
            except Exception as e:
                print(e)

    return _reader


if __name__ == '__main__':
    tmp = reader("./data/f", im_size=[256, 256])
    for i in tmp():
        print(i)
