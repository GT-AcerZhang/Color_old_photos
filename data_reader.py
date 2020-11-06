import os
import random
import traceback

import paddle.io
import numpy as np
import cv2 as cv

from color1d import cvt2label, cvt2color

DICT_FILE_PATH = "./color_files/Color1D_Base.dict"  # 颜色空间文件
RESIZE = 256

# 读取颜色空间字典
with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
    dict_info = eval(f.read())
    label_map = dict_info["2label"]
    weight = dict_info["weight"]


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


def cvt_sample_with_lab(sample):
    re_img = cv.resize(sample, (RESIZE, RESIZE))
    lab_img = cv.cvtColor(re_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_img)
    a_label = cvt2label(a, label_map[0])
    b_label = cvt2label(b, label_map[1])
    l_array = np.array([l], dtype="float32") / 255
    a_array = np.array([a_label], dtype="int64")
    b_array = np.array([b_label], dtype="int64")
    ab_array = np.concatenate([a_array, b_array], axis=0)
    return l_array, ab_array


def cvt_sample_with_l(sample):
    re_img = cv.resize(sample, (RESIZE, RESIZE))
    l = cv.cvtColor(re_img, cv.COLOR_BGR2GRAY)
    l_array = np.array([l], dtype="float32") / 255
    # l_array = np.expand_dims(l_array, axis=0)
    return l_array


class Reader(paddle.io.Dataset):
    def __init__(self, data_dir, add_label: bool = True, cache_file="./cache.list"):
        super().__init__()
        self.add_label = add_label
        img_list = os.listdir(data_dir)
        self.img_list = list()

        if not add_label:
            cache_file += "test"
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as cache:
                print("读取到文件列表cache，将使用cache中包含的文件。如修改数据集请指定Reader参数use_cache为False")
                try:
                    self.img_list = eval(cache.read())
                except Exception as e:
                    print("ERROR:", e)
                    raise Exception("已终止，数据读取异常！")
        else:
            print("正在检查可用数据...")
            bad_count = 0
            for file_id, file in enumerate(img_list):
                if file_id == 1000 and not add_label:
                    break
                if os.path.splitext(file)[1] not in [".jpg", ".jpeg", ".bmp", "png"]:
                    bad_count += 1
                    continue
                file_path = os.path.join(data_dir, file)
                try:
                    img = cv.imread(file_path)
                    img = cv.resize(img, (128, 128))
                    if check_gray(img) and add_label:
                        bad_count += 1
                        continue
                    else:
                        self.img_list.append(file_path)
                except Exception as e:
                    bad_count += 1
                    print("ERROR", e)
                print("ERROR COUNT:", bad_count, "/", len(self.img_list),
                      "\t{:.4f}%".format((len(self.img_list) + bad_count) / len(img_list) * 100), end="\r")
            with open(cache_file, "w", encoding="utf-8") as cache:
                cache.write(str(self.img_list))
            print("可用数据：", len(self.img_list), "异常数据：", bad_count)

    def __getitem__(self, index):
        file_path = self.img_list[index]
        try:
            img = cv.imread(file_path)
        except Exception as e:
            print("ERROR:", e)
            raise Exception("已终止，数据读取异常！")

        if self.add_label:
            l, ab = cvt_sample_with_lab(img)
            return l, ab
        else:
            return cvt_sample_with_l(img)

    def __len__(self):
        return len(self.img_list)


def get_weight():
    w_a = np.array(weight[0]).astype("float32")
    w_b = np.array(weight[1]).astype("float32")
    return w_a, w_b


def get_class_num():
    return len(weight[0]), len(weight[1])


if __name__ == '__main__':
    import time

    tmp = Reader("data/ffff", add_label=True)

    with open(DICT_FILE_PATH, "r", encoding="utf-8") as f:
        color_map = eval(f.read())["2color"]


    def vdl_1d(l_vdl, a_vdl, b_vdl, name):
        a_vdl = cvt2color(a_vdl, color_map[0])
        b_vdl = cvt2color(b_vdl, color_map[1])
        l_vdl = l_vdl.astype("uint8")
        tmp_img = np.concatenate([l_vdl, a_vdl, b_vdl], axis=2)
        tmp_img = cv.cvtColor(tmp_img, cv.COLOR_LAB2BGR)
        cv.imshow(name, tmp_img)


    def vdl_2d(l_vdl, a_vdl, b_vdl, name):
        a_vdl = cvt2color(a_vdl, color_map[0])
        b_vdl = cvt2color(b_vdl, color_map[1])
        tmp_img = cv.merge([l_vdl.astype("uint8"), a_vdl, b_vdl])
        tmp_img = cv.cvtColor(tmp_img, cv.COLOR_LAB2BGR)
        cv.imshow(name, tmp_img)


    tmp_c = get_class_num()
    tmp_w = get_weight()
    print("Class_num", tmp_c)
    print("Weight", tmp_w)

    start = time.time()
    for inx, i in enumerate(tmp):
        vdl_1d(i[0][0] * 256, i[0][1][0], i[0][1][1], "scale")
        cv.waitKey(0)
