# Author: Acer Zhang
# Datetime:2020/7/6 21:39
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import numpy as np
import cv2 as cv
import paddle.fluid as fluid

from data_reader import reader

TEST_DATA_PATH = "./data/val"
MODEL_DIR = "./best_model.color"
DICT_PATH = "./Color.dict"
IM_SIZE = [256] * 2


def cvt_color(ipt, c_map):
    f_a = np.zeros_like(ipt)
    f_b = np.zeros_like(ipt)
    for x in range(IM_SIZE[0]):
        for y in range(IM_SIZE[1]):
            f_a[x][y], f_b[x][y] = c_map[ipt[x][y]]
    return f_a, f_b


with open(DICT_PATH, "r", encoding="utf-8") as f:
    color_map = eval(f.read())[1]

place = fluid.CPUPlace()
exe = fluid.Executor(place)

infer_reader = fluid.io.batch(
    reader=reader(TEST_DATA_PATH, is_val=True, im_size=IM_SIZE),
    batch_size=1)

program, feed_list, target_list = fluid.io.load_inference_model(dirname=MODEL_DIR, executor=exe)
feeder = fluid.DataFeeder(place=place, feed_list=feed_list, program=program)

for data in infer_reader():
    ipt_data = [i[0] for i in data]
    ipt_l = [i[1] for i in data]
    ipt_h = [i[2] for i in data]
    ipt_w = [i[3] for i in data]
    out = exe.run(program, feeder.feed(ipt_data), fetch_list=target_list)
    for img_h, img_w, img_l, img_ab in zip(ipt_h, ipt_w, ipt_l, out[0]):
        img_ab = img_ab.reshape((IM_SIZE[0], IM_SIZE[1]))*300
        img_a, img_b = cvt_color(img_ab.astype("uint8"), color_map)
        img_a_r = cv.resize(img_a.astype("uint8"), (img_w, img_h))
        img_b_r = cv.resize(img_b.astype("uint8"), (img_w, img_h))
        img = cv.merge([img_l, img_a_r, img_b_r])
        im = cv.cvtColor(img, cv.COLOR_LAB2BGR)
        # im = cv.resize(im, (img_w, img_h))
        cv.imshow("Result", im)
        cv.imwrite("result.jpg", im)
        cv.waitKey()
