# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import paddle
from paddle.static import InputSpec

from color_net import ColorNet
from data_reader import Reader, get_weight, get_class_num, get_resize

LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False
DEBUG = False

ROOT_PATH = "./"
DEBUG_PATH = os.path.join(ROOT_PATH, "DEBUG")
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, "data/train") if not DEBUG else DEBUG_PATH
EVAL_DATA_PATH = os.path.join(ROOT_PATH, "data/eval") if not DEBUG else DEBUG_PATH
CHECK_POINT_DIR = os.path.join(ROOT_PATH, "check_point/check_model.color")
PER_MODEL_DIR = os.path.join(ROOT_PATH, "data/unet_coco_v3")

EPOCH = 30 if not DEBUG else 1000
BATCH_SIZE = 16 if not DEBUG else 8
MAX_ITER_NUM = 100000
L1_MODE = False

BOUNDARIES = [50000, 100000, 500000, 1000000]
VALUES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
WARM_UP_STEPS = int(1000 / BATCH_SIZE) if not DEBUG else 10
START_LR = 0.01
END_LR = 0.05
GLOBAL_STEP = 0

RESIZE = get_resize()
CLASS_NUM_A, CLASS_NUM_B = get_class_num()


class ColorCrossEntropy(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, l1, l2, ab2, lab):
        # 计算L1与L2损失
        loss_l = paddle.nn.functional.smooth_l1_loss(l2, l1)

        # 计算AB2与Label损失
        ab2 = paddle.tensor.transpose(ab2, [0, 2, 3, 1])
        lab = paddle.tensor.transpose(lab, [0, 2, 3, 1])
        if L1_MODE:
            y_a, y_b = paddle.tensor.split(ab2, 2, axis=3)
            lab_a, lab_b = paddle.tensor.split(lab, 2, axis=3)
            lab_a = paddle.tensor.cast(lab_a, "float32")
            lab_b = paddle.tensor.cast(lab_b, "float32")
            lab_a /= CLASS_NUM_A
            lab_b /= CLASS_NUM_B
            loss_a = paddle.nn.functional.smooth_l1_loss(y_a, lab_a)
            loss_b = paddle.nn.functional.smooth_l1_loss(y_b, lab_b)
            return loss_l + loss_a + loss_b
        else:
            y_a, y_b = paddle.tensor.split(ab2, [CLASS_NUM_A, CLASS_NUM_B], axis=3)
            lab_a, lab_b = paddle.tensor.split(lab, 2, axis=3)
            loss_a = paddle.nn.functional.softmax_with_cross_entropy(y_a,
                                                                     lab_a,
                                                                     return_softmax=False,
                                                                     axis=3)
            loss_b = paddle.nn.functional.softmax_with_cross_entropy(y_b,
                                                                     lab_b,
                                                                     return_softmax=False,
                                                                     axis=3)
            return paddle.mean(loss_a + loss_b) + loss_l


inputs = [InputSpec(shape=[-1, 1, RESIZE, RESIZE], name="GARY_AB")]
labels = [InputSpec(shape=[-1, 2, RESIZE, RESIZE], dtype="int64", name="LABEL_AB")]
net = ColorNet(CLASS_NUM_A + CLASS_NUM_B) if not L1_MODE else ColorNet(2)
model = paddle.Model(net, inputs, labels)
optimizer = paddle.optimizer.Adam(0.001, parameters=model.parameters())

train_reader = Reader(TRAIN_DATA_PATH)
eval_reader = Reader(EVAL_DATA_PATH)

if LOAD_CHECKPOINT:
    model.load(path=os.path.join(CHECK_POINT_DIR, "final"), skip_mismatch=True)
model.prepare(optimizer, loss=ColorCrossEntropy())
model.fit(train_data=train_reader,
          eval_data=eval_reader,
          batch_size=BATCH_SIZE,
          epochs=EPOCH,
          log_freq=500,
          eval_freq=200,
          save_dir="./checkpoints_hapi" if not DEBUG else "./DEBUG_CHECK",
          save_freq=1 if not DEBUG else 1000)
