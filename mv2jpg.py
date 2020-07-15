import os

import cv2 as cv

PATH = './data/test.mov'
SAVE_PATH = "./data/out_put"

print("Load...")
cap = cv.VideoCapture(PATH)
frame_id = 0
cap_act = cap.isOpened()
print("Save...")
while cap_act:
    frame_id += 1
    cap_act, frame = cap.read()
    if frame_id % 20 == 0:
        cv.imwrite(os.path.join(SAVE_PATH, str(frame_id) + ".bmp"), frame, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        print(frame_id, "OK!")
print("ALL OK!")
