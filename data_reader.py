# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import numpy as np
import PIL.Image as Image


def reader(data_path, is_val: bool = False, im_size: list = None):
    file_names = os.listdir(data_path)

    def _reader():
        for file_name in file_names:
            try:
                if is_val:
                    ori_img = Image.open(os.path.join(data_path, file_name)).convert("L")
                    ori_w, ori_h = ori_img.size
                    im_l = ori_img.resize(im_size)
                    im_l = np.array(im_l).reshape((1, 1, im_size[0], im_size[1])).astype("float32")
                    ori_l = np.array(ori_img).reshape((ori_h, ori_w)).astype("uint8")
                    yield im_l / 255, ori_l, ori_h, ori_w
                else:
                    ori_img = np.load(os.path.join(data_path, file_name))
                    ori_h, ori_w = ori_img.shape[1:]
                    l, label = ori_img[0, :, :], ori_img[1, :, :]
                    im_l = np.array(l).reshape((1, 1, ori_h, ori_w)).astype("float32")
                    im_label = np.array(label).reshape((1, 1, ori_h, ori_w)).astype("float32")
                    yield im_l / 255, im_label / 300
            except Exception as e:
                print(e)

    return _reader


if __name__ == '__main__':
    tmp = reader("./data/tmp")
    for i in tmp():
        print(i)
