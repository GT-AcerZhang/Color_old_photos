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

IM_SIZE = [256] * 2

place = fluid.CPUPlace()
exe = fluid.Executor(place)

infer_reader = fluid.io.batch(
    reader=reader(TEST_DATA_PATH, IM_SIZE, is_val=True),
    batch_size=1)

program, feed_list, target_list = fluid.io.load_inference_model(dirname=MODEL_DIR, executor=exe)
feeder = fluid.DataFeeder(place=place, feed_list=feed_list, program=program)

for data in infer_reader():
    ipt_data = [i[0] for i in data]
    ipt_h = [i[1] for i in data]
    ipt_w = [i[2] for i in data]
    out = exe.run(program, feeder.feed(ipt_data), fetch_list=target_list)
    for img_h, img_w, img_l, img_ab in zip(ipt_h, ipt_w, ipt_data, out[0]):
        img_l = img_l.reshape((1, 512, 512))
        img = np.concatenate([img_l, img_ab]).reshape((3, 512, 512)).transpose(1, 2, 0) * 254
        im = cv.cvtColor(img.astype("uint8"), cv.COLOR_LAB2BGR)
        im = cv.resize(im, (img_w, img_h))
        cv.imshow("Result", im)
        cv.imwrite("result.jpg", im)
        cv.waitKey()
