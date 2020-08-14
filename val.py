# Author: Acer Zhang
# Datetime:2020/7/6 21:39
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import numpy as np
import cv2 as cv
import paddle.fluid as fluid
from matplotlib import pyplot as plt

from data_reader import reader
from color2d import cvt2color

TEST_DATA_PATH = "./data/fff"
MODEL_DIR = "./color.model"
DICT_PATH = "./color_files/Color_2D.dict"

with open(DICT_PATH, "r", encoding="utf-8") as f:
    color_map = eval(f.read())["2color"]


def vdl(l_vdl, ab_vdl, name):
    ab_vdl = cvt2color(ab_vdl, color_map)
    l_vdl = np.expand_dims(l_vdl, axis=2).astype("uint8")
    tmp_img = np.concatenate([l_vdl, ab_vdl], axis=2)
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
    signal_l, signal_ab = out
    vdl(signal_l[0][0] * 128 + 128, signal_ab[0], "all")
    vdl(ori_l[0][0] * 128 + 128, signal_ab[0], "ori_l")
    cv.waitKey(0)
