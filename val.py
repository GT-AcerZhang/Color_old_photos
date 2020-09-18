# Author: Acer Zhang
# Datetime:2020/7/6 21:39
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import numpy as np
import cv2 as cv
import paddle.fluid as fluid
from matplotlib import pyplot as plt

from data_reader import reader
# from color2d import cvt2color
from color1d import cvt2color

TEST_DATA_PATH = "./data/ff"
MODEL_DIR = "./color.model"
DICT_PATH = "./color_files/Color1D_Base.dict"

with open(DICT_PATH, "r", encoding="utf-8") as f:
    color_map = eval(f.read())["2color"]


def vdl(l_vdl, a_vdl, b_vdl, name):
    a_vdl = cvt2color(a_vdl, color_map[0])
    b_vdl = cvt2color(b_vdl, color_map[1])
    a_vdl = np.expand_dims(a_vdl, axis=2).astype("uint8")
    b_vdl = np.expand_dims(b_vdl, axis=2).astype("uint8")
    l_vdl = np.expand_dims(l_vdl, axis=2).astype("uint8")
    tmp_img = np.concatenate([l_vdl, a_vdl, b_vdl], axis=2)
    tmp_img = cv.cvtColor(tmp_img, cv.COLOR_LAB2BGR)
    cv.imshow(name, tmp_img)


place = fluid.CPUPlace()
exe = fluid.Executor(place)

reader = reader(TEST_DATA_PATH, is_infer=True)

program, feed_list, target_list = fluid.io.load_inference_model(dirname=MODEL_DIR, executor=exe)

for data in reader():
    ori_l = data[0]
    ori_r = data[1]
    print("Done")
    out = exe.run(program, feed={feed_list[0]: ori_l, feed_list[1]: ori_r}, fetch_list=target_list)
    signal_l, signal_a, signal_b = out
    vdl(signal_l[0][0] * 128 + 128, signal_a[0], signal_b[0], "all")
    vdl(ori_l[0][0] * 128 + 128, signal_a[0], signal_b[0], "ori")
    cv.imshow("ori_l", ori_l[0][0])
    cv.waitKey(0)
