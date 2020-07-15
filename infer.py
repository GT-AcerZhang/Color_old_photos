# Author: Acer Zhang
# Datetime:2020/7/6 21:39
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import cv2 as cv
import paddle.fluid as fluid
from matplotlib import pyplot as plt

from data_reader import reader
from cvt_image_data import cvt_color

TEST_DATA_PATH = "./data/out_put"
MODEL_DIR = "./best_model.color"
DICT_PATH = "./Color1D.dict"

with open(DICT_PATH, "r", encoding="utf-8") as f:
    a_dict, b_dict = eval(f.read())[1]
IM_SIZE = [256] * 2

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
        img_a_b = img_ab.reshape((2, IM_SIZE[0], IM_SIZE[1])).transpose(1, 2, 0).astype("uint8")
        img_a, img_b = cv.split(img_a_b)
        img_a = cvt_color(img_a, a_dict)
        img_b = cvt_color(img_b, b_dict)
        img_a_r = cv.resize(img_a, (img_w, img_h), interpolation=cv.INTER_NEAREST)
        img_b_r = cv.resize(img_b, (img_w, img_h), interpolation=cv.INTER_NEAREST)
        img = cv.merge([img_l, img_a_r, img_b_r])
        im = cv.cvtColor(img, cv.COLOR_LAB2BGR)
        im = cv.resize(im, (img_w, img_h))
        plt.imshow(im)
        plt.show()
