import os
import traceback
import random

import paddle.fluid as fluid
import numpy as np
import cv2 as cv

CPU_NUM = 4
DICT_FILE_PATH = "./color_files/Color1D_Base_v2.dict"
with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
    c_dict = eval(f.read())[0]


def check_gray(ipt):
    img_hsv = cv.cvtColor(ipt, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_hsv)
    s_w, s_h = s.shape[:2]
    s_sum = np.sum(s) / (s_w * s_h)
    if s_sum > 15:
        return False
    else:
        return True


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


def make_train_data(sample):
    try:
        r_scale = random.uniform(0.25, 0.8)
        # 此处可以添加数据增强代码
        pre_done_img = sample
        pre_done_img = cv.cvtColor(pre_done_img, cv.COLOR_BGR2LAB)
        cvt_l, cvt_a, cvt_b = cvt_process(pre_done_img, c_dict)

        cvt_l_label = cv.resize(cvt_l, (
            int(pre_done_img.shape[1] * r_scale), int(pre_done_img.shape[0] * r_scale)))
        cvt_l_label = cv.resize(cvt_l_label, (pre_done_img.shape[1], pre_done_img.shape[0]))

        cvt_l_label = np.expand_dims(np.array(cvt_l_label), 0).astype("float32")
        cvt_l = np.expand_dims(np.array(cvt_l), 0).astype("float32")

        cvt_a = np.expand_dims(np.array(cvt_a), 0).astype("int64")
        cvt_b = np.expand_dims(np.array(cvt_b), 0).astype("int64")
        im_shape = np.array(pre_done_img.shape[:-1]).astype("int32")

        return im_shape, cvt_l_label / 255, cvt_l / 255, cvt_a, cvt_b

    except Exception as e:
        traceback.print_exc()


def reader(data_path, is_val: bool = False, debug: bool = False):
    file_names = os.listdir(data_path)

    def _reader():
        for file_name in file_names:
            if os.path.splitext(file_name)[1] not in [".jpg", ".jpeg", ".bmp", "png"]:
                print(file_name, "is not right img file, so skip this item!")
                continue
            if is_val:
                pass
            else:
                ori_img = cv.imread(os.path.join(data_path, file_name))
                check_im = cv.resize(ori_img, (32, 32))
                if check_gray(check_im):
                    print(file_name, "like L mode, so skip it")
                    continue
                else:
                    yield ori_img

    return fluid.io.xmap_readers(make_train_data, _reader, CPU_NUM, 64)


if __name__ == '__main__':
    tmp = reader("./data/f", debug=True)
    for i in tmp():
        print(1)
