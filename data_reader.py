import os
import random
import traceback

import paddle.fluid as fluid
import numpy as np
import cv2 as cv

from color2d import cvt2label, cvt2color

DEBUG = False
CPU_NUM = 4  # CPU 队列数 不推荐过高
MAX_BATCH_SIZE = 1  # BATCH SIZE 阈值，16G显存推荐为2
MEMORY_CAPACITY = 16.0  # 硬件会保留部分显存，此处为可用内存大小，单位GB
DICT_FILE_PATH = "./color_files/Color_2D.dict"  # 颜色空间文件
RESIZE = 256

# 读取颜色空间字典
with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
    dict_info = eval(f.read())
    label_map = dict_info["2label"]
    weight = dict_info["weight"]

# 判断Mini Batch数据大小以及图片缩放系数
if MEMORY_CAPACITY // 16 == 0:
    SAMPLE_NUM = 1
    RAM_SCALE = 0.8
else:
    SAMPLE_NUM = int(MEMORY_CAPACITY // 16)
    RAM_SCALE = 1.


def get_resize():
    return RESIZE


def check_gray(ipt):
    """
    检查是否为可疑的灰度图像
    :param ipt: opencv图像对象
    :return: 布尔判断结果
    """
    img_hsv = cv.cvtColor(ipt, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_hsv)
    s_w, s_h = s.shape[:2]
    s_sum = np.sum(s) / (s_w * s_h)
    if s_sum > 15:
        return False
    else:
        return True


def make_train_data(sample):
    sample, is_test, freeze_pix = sample
    tmp_cvt_l_label = []
    tmp_cvt_l = []
    if is_test:
        sample_num = 1
    else:
        sample_num = SAMPLE_NUM
    for index in range(sample_num):
        # 原始图像随机尺寸缩放
        if is_test:
            r_ori_scale = 0.9
            r_scale = 0.5
        else:
            r_ori_scale = random.uniform(0.25, 0.9)
            r_scale = random.uniform(0.5, 0.8)
        sample_h, sample_w = sample.shape[:2]
        sample = cv.resize(sample, (int(sample_w * RAM_SCALE), int(sample_h * RAM_SCALE)))
        sample_h, sample_w = sample.shape[:2]
        pre_done_img = cv.resize(sample, (
            int(sample_w * r_ori_scale), int(sample_h * r_ori_scale)))
        pre_done_img = cv.resize(pre_done_img, (sample_w, sample_h))

        # 转化颜色空间
        pre_done_img = cv.cvtColor(pre_done_img, cv.COLOR_BGR2LAB)

        # 压缩颜色空间
        cvt_l, cvt_ab = pre_done_img[:, :, 0], cvt2label(pre_done_img[:, :, 1:], label_map)
        cvt_l_fix_shape = cv.resize(cvt_l, (RESIZE, RESIZE), interpolation=cv.INTER_NEAREST)
        cvt_ab = cv.resize(cvt_ab, (RESIZE, RESIZE), interpolation=cv.INTER_NEAREST)

        # 生成模糊图像
        cvt_l_label = cv.resize(cvt_l,
                                (int(pre_done_img.shape[1] * r_scale), int(pre_done_img.shape[0] * r_scale)),
                                interpolation=cv.INTER_NEAREST)
        cvt_l_label = cv.resize(cvt_l_label,
                                (pre_done_img.shape[1], pre_done_img.shape[0]),
                                interpolation=cv.INTER_NEAREST)

        # 数据增强 - 翻转
        for mode in random.sample([-1, 0, 1, -2], 4):
            if freeze_pix == "AB":
                return (np.array([cvt_l_fix_shape]).astype("float32") - 128) / 128, \
                       np.array([cvt_ab]).astype("int64")

            if mode == -2 or is_test:
                tmp_cvt_l_label.append([cvt_l_label])
                tmp_cvt_l.append([cvt_l])
                if is_test:
                    break
            else:
                tmp_cvt_l_label.append([cv.flip(cvt_l_label, mode)])
                tmp_cvt_l.append([cv.flip(cvt_l, mode)])

    cvt_l_label = np.array(tmp_cvt_l_label).astype("float32")
    cvt_l = np.array(tmp_cvt_l).astype("float32")
    pack = []

    for index in range(int(SAMPLE_NUM * 4)):
        if index == MAX_BATCH_SIZE:
            break
        pack_t = ((cvt_l_label[index] - 128) / 128,
                  (cvt_l[index] - 128) / 128)
        pack.append(pack_t)
        if is_test:
            break
    return pack


def reader(data_path, is_test: bool = False, is_infer: bool = False, freeze_pix="L"):
    file_names = os.listdir(data_path)

    def _reader():
        for file_name in file_names:
            if os.path.splitext(file_name)[1] not in [".jpg", ".jpeg", ".bmp", "png"]:
                print(file_name, "is not right img file, so skip this item!")
                continue
            if is_infer:
                try:
                    ori_img = cv.imread(os.path.join(data_path, file_name))
                    l_img = cv.cvtColor(ori_img, cv.COLOR_BGR2LAB)
                    l_img, _, _ = cv.split(l_img)
                    r_img = cv.resize(l_img, (RESIZE, RESIZE))
                    l_img = np.array([[l_img]]).astype("float32") / 255
                    r_img = np.array([[r_img]]).astype("float32") / 255
                    yield l_img, r_img
                except Exception as e:
                    print(file_name, "Image reading failed!", e)
                    traceback.print_exc()
            else:
                ori_img = cv.imread(os.path.join(data_path, file_name))
                check_im = cv.resize(ori_img, (32, 32))
                if check_gray(check_im):
                    if DEBUG:
                        print(file_name, "like L mode, so skip it")
                    continue
                else:
                    yield ori_img, is_test, freeze_pix

    return fluid.io.xmap_readers(make_train_data, _reader, CPU_NUM, CPU_NUM * 4) if not is_infer else _reader


def get_weight():
    w = np.array(weight).astype("float32")
    return w


def get_class_num():
    return len(weight)


if __name__ == '__main__':
    tmp = reader("data/ff", freeze_pix="AB")
    with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
        color_map = eval(f.read())["2color"]


    def vdl(l_vdl, ab_vdl, name):
        ab_vdl = cvt2color(ab_vdl, color_map)
        l_vdl = np.expand_dims(l_vdl, axis=2).astype("uint8")
        tmp_img = np.concatenate([l_vdl, ab_vdl], axis=2)
        tmp_img = cv.cvtColor(tmp_img, cv.COLOR_LAB2BGR)
        cv.imshow(name, tmp_img)


    tmp_c = get_class_num()
    tmp_w = get_weight()
    print("Class_num", tmp_c)
    print("Weight", tmp_w)
    for i in tmp():
        if i:
            print("OK")
            vdl(i[0][0] * 128 + 128, i[1][0], "scale")
            cv.waitKey(0)
