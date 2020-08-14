# Author: Acer Zhang
# Datetime:2020/7/6 21:39
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import cv2 as cv
import paddle.fluid as fluid
from matplotlib import pyplot as plt

from data_reader import reader
from data_reader import cvt_color

TEST_DATA_PATH = "./data/ff"
MODEL_DIR = "./color.model"
DICT_PATH = "./color_files/Color1D_Beta2.dict"

with open(DICT_PATH, "r", encoding="utf-8") as f:
    a_dict, b_dict = eval(f.read())["2ori"]


def visual_img(l, a, b, name):
    a = cvt_color(a, a_dict)
    b = cvt_color(b, b_dict)
    tmp_img = cv.merge([l.astype("uint8"), a.astype("uint8"), b.astype("uint8")])
    tmp_img = cv.cvtColor(tmp_img, cv.COLOR_LAB2BGR)
    cv.imshow(name, tmp_img)


place = fluid.CPUPlace()
exe = fluid.Executor(place)

reader = reader(TEST_DATA_PATH, is_infer=True)

program, feed_list, target_list = fluid.io.load_inference_model(dirname=MODEL_DIR, executor=exe)

for data in reader():
    ori_l = data[0]
    ori_r = data[1]
    out = exe.run(program, feed={feed_list[0]: ori_l, feed_list[1]: ori_r}, fetch_list=target_list)
    signal_l, signal_a_out, signal_b_out = out
    visual_img(signal_l[0][0] * 128 + 128, signal_a_out[0], signal_b_out[0], "all")
    visual_img(ori_l[0][0] * 128 + 128, signal_a_out[0], signal_b_out[0], "ori_l")
    cv.waitKey(0)
