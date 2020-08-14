import cv2 as cv

DICT_PATH = "color_files/Color1D_Beta.dict"
IMAGE = r"/Users/zhanghongji/PycharmProjects/Color_old_photos/data/ff/000000000275.jpg"


def cvt_color(ori_img, color_dict: dict):
    h, w = ori_img.shape
    for pix_h in range(h):
        for pix_w in range(w):
            ori_img[pix_h][pix_w] = color_dict[ori_img[pix_h][pix_w]]
    return ori_img


def cvt_process(ori_img, color_dict):
    a_dict, b_dict = color_dict
    r_img = cv.cvtColor(ori_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(r_img)
    l_a = cvt_color(a, a_dict)
    l_b = cvt_color(b, b_dict)
    return l, l_a, l_b


with open(DICT_PATH, "r", encoding="utf-8") as f:
    info = f.read()
    c_d = eval(info)['2mini']
    a_dict1, b_dict1 = eval(info)["2ori"]

IM_SIZE = [256] * 2

tmp = cv.imread(IMAGE)  # 83382486
o_l, label_a, label_b = cvt_process(tmp, c_d)
img_a = cvt_color(label_a, a_dict1)
img_b = cvt_color(label_b, b_dict1)
img = cv.merge([o_l, img_a, img_b])
im = cv.cvtColor(img, cv.COLOR_LAB2BGR)
cv.imshow("test", im)
cv.imshow("ori", tmp)
cv.waitKey(0)
