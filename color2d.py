import os

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

CLASS_NUM = 32
MAX_ITER = 10
R_IMG_SIZE = (512, 512)  # 采样分辨率，越高越好 - 可能出现内存爆炸现象
IMAGE_DIR_PATH = "./test2017"
SAVE_PATH = "color_files/Color_2D"


def cvt2label(ab_img, label_map):
    h, w = ab_img.shape[:2]
    result = np.zeros([h, w], dtype="uint8")
    for pix_h in range(h):
        for pix_w in range(w):
            pix = ab_img[pix_h][pix_w]
            result[pix_h][pix_w] = label_map[pix[0]][pix[1]]
    return result


def cvt2color(label_img, color_map):
    h, w = label_img.shape[:2]
    label_img = label_img.astype("uint8")
    result = np.zeros([h, w, 2], dtype="uint8")
    for pix_h in range(h):
        for pix_w in range(w):
            pix = label_img[pix_h][pix_w]
            result[pix_h][pix_w][0], result[pix_h][pix_w][1] = color_map[pix]
    return result


def analysis(image_dir_path, class_num):
    k_means = KMeans(n_clusters=class_num, max_iter=10, n_init=20)
    cache = None
    label_map = None
    color_map = None
    for file_id, file in enumerate(os.listdir(image_dir_path)):
        if os.path.splitext(file)[1] not in [".jpg", ".jpeg", ".bmp"]:
            continue
        img = cv.imread(os.path.join(IMAGE_DIR_PATH, file))
        img = cv.resize(img, R_IMG_SIZE)
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        # 取AB通道
        img = np.array(img).reshape([-1, 3])[:, 1:]
        if file_id % 5 == 4:
            k_means.fit(cache)
            cache = None
        elif cache is None:
            cache = img
        else:
            cache = np.concatenate([cache, img], axis=0)
        if file_id % 10 == 0:
            print(file_id, "Done")
        if file_id == MAX_ITER - 1:
            label_map = k_means.predict([[a, b] for a in range(256) for b in range(256)]).reshape([256, 256])
            color_map = k_means.cluster_centers_
            break

    print("Analysis weight...")
    img_h = None
    for file_id, file in enumerate(os.listdir(image_dir_path)):
        if os.path.splitext(file)[1] not in [".jpg", ".jpeg", ".bmp"]:
            continue
        img = cv.imread(os.path.join(IMAGE_DIR_PATH, file))
        img = cv.resize(img, R_IMG_SIZE)
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        img = np.array(img)[:, :, 1:]
        img = cvt2label(img, label_map).flatten()
        if img_h is None:
            img_h = np.histogram(img, CLASS_NUM, range=(0, CLASS_NUM))[0].astype("float32")
        else:
            img_h += np.histogram(img, CLASS_NUM, range=(0, CLASS_NUM))[0].astype("float32")
            img_h //= 2
        if file_id == MAX_ITER - 1:
            break
    img_w = 1 - (img_h.astype("float32") / np.sum(img_h))
    print("Weight Done")
    return label_map.astype("int64"), color_map.astype("int64"), img_w


if __name__ == '__main__':
    label_m, color_m, weight = analysis(IMAGE_DIR_PATH, CLASS_NUM)
    with open(SAVE_PATH + ".dict", "w", encoding="utf-8") as f:
        f.write(str({"2label": label_m.tolist(), "2color": color_m.tolist(), "weight": weight.tolist()}))
    with open(SAVE_PATH + ".info", "w", encoding="utf-8") as f:
        f.write("CLASS_NUM: " + str(CLASS_NUM) + "\nR_IMG_SIZE: " + str(R_IMG_SIZE) + "\nMAX_ITER: " + str(MAX_ITER))
    im = cv.imread("./data/f/000000168337.jpg")
    im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
    c_im = im[:, :, 1:]
    c_im = cvt2label(c_im, label_m.tolist())
    c_im = cvt2color(c_im, color_m.tolist())
    im2 = cv.cvtColor(im, cv.COLOR_LAB2BGR)
    cv.imshow("ori", im2)
    im2 = np.concatenate([im[:, :, :1], c_im], axis=2)
    im2 = cv.cvtColor(im2, cv.COLOR_LAB2BGR)
    cv.imshow("cvt", im2)
    cv.waitKey(0)
