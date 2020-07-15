import os

import numpy as np
import PIL.Image as Image  # 随后转为cv2


def req_color_map(im):
    count = np.histogram(im, 256, range=(0, 256))[0].astype("float32")
    count[count == 0] = 1.
    count_t = np.reciprocal(count)
    color_map = dict([(k, v) for k, v in zip(range(256), count_t)])
    return color_map


def req_weight(im, color_map):
    """
    获取权重
    :param im: 图像数据 - 2维
    :param color_map: 颜色权重表
    :return: 用于损失函数加权的数据
    """
    im = im.copy()
    for pix_value in range(256):
        if pix_value in im:
            im[im == pix_value] = color_map[pix_value]
    return im


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
                    ori_h, ori_w = ori_img.shape[:-1]
                    ori_img = np.array(ori_img).transpose([2, 0, 1])
                    l, a, b = ori_img[0, :, :], ori_img[1, :, :], ori_img[2, :, :]
                    im_l = np.array(l).reshape((1, 1, ori_h, ori_w)).astype("float32")
                    im_a = np.array(a).reshape((1, 1, ori_h, ori_w)).astype("int64")
                    im_b = np.array(b).reshape((1, 1, ori_h, ori_w)).astype("int64")
                    a_color_map = req_color_map(im_a)
                    b_color_map = req_color_map(im_b)
                    a_w = req_weight(im_a, a_color_map)
                    b_w = req_weight(im_b, b_color_map)
                    yield im_l / 255, im_a, im_b, a_w, b_w
            except Exception as e:
                print(e)

    return _reader


if __name__ == '__main__':
    tmp = reader("./data/ff", im_size=[256, 256])
    for i in tmp():
        print(i)
