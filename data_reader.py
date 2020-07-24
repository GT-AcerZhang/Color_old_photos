import os
import traceback
import random

import paddle.fluid as fluid
import numpy as np
import cv2 as cv

CPU_NUM = 4
GPU_NUM = 8
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
    tmp_cvt_l_label = []
    tmp_cvt_l = []
    tmp_cvt_a = []
    tmp_cvt_b = []
    for _ in range(GPU_NUM):
        # 原始图像随机尺寸缩放
        r_ori_scale = random.uniform(0.25, 1.)
        r_scale = random.uniform(0.5, 0.8)
        sample_h, sample_w = sample.shape[:2]
        pre_done_img = cv.resize(sample, (
            int(sample_w * r_ori_scale), int(sample_h * r_ori_scale)))
        pre_done_img = cv.resize(pre_done_img, (sample_w, sample_h))

        # 转化颜色空间
        pre_done_img = cv.cvtColor(pre_done_img, cv.COLOR_BGR2LAB)

        # 压缩颜色空间
        cvt_l, cvt_a, cvt_b = cvt_process(pre_done_img, c_dict)

        # 生成低分辨率图像
        cvt_l_label = cv.resize(cvt_l, (
            int(pre_done_img.shape[1] * r_scale), int(pre_done_img.shape[0] * r_scale)))
        cvt_l_label = cv.resize(cvt_l_label, (pre_done_img.shape[1], pre_done_img.shape[0]))

        # 数据翻转增强
        for mode in random.sample([-1, 0, 1], 3):
            tmp_cvt_l_label.append([cv.flip(cvt_l_label, mode)])
            tmp_cvt_l.append([cv.flip(cvt_l, mode)])
            tmp_cvt_a.append([cv.flip(cvt_a, mode)])
            tmp_cvt_b.append([cv.flip(cvt_b, mode)])
        tmp_cvt_l_label.append([cvt_l_label])
        tmp_cvt_l.append([cvt_l])
        tmp_cvt_a.append([cvt_a])
        tmp_cvt_b.append([cvt_b])
    cvt_l_label = np.array(tmp_cvt_l_label).astype("float32")
    cvt_l = np.array(tmp_cvt_l).astype("float32")
    cvt_a = np.array(tmp_cvt_a).astype("int64")
    cvt_b = np.array(tmp_cvt_b).astype("int64")
    return cvt_l_label / 255, cvt_l / 255, cvt_a, cvt_b


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
