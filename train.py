# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import paddle.fluid as fluid

from models.libs.model_libs import scope
from models.modeling.unet import encode, decode, get_logit, deconv, double_conv
from data_reader import reader
from net import get_net, up

LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False

TRAIN_DATA_PATH = "./data/train"
TEST_DATA_PATH = "./data/test"
CHECK_POINT_DIR = "./check_point/check_model.color"
PER_MODEL_DIR = "./data/unet_coco_v3"
MODEL_DIR = "./best_model.color"

EPOCH = 500
BATCH_SIZE = 8
IM_SIZE = [256] * 2
OUT_IM_SIZE = [IM_SIZE[0] * 2, IM_SIZE[1] * 2]
SIGNAL_A_NUM = 14
SIGNAL_B_NUM = 14

BOUNDARIES = [1000, 5000, 15000, 100000]
VALUES = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
WARM_UP_STEPS = 150
START_LR = 0.005
END_LR = 0.01

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    img_l = fluid.data(name="img_l", shape=[-1, 1] + IM_SIZE)
    o_img_l = fluid.data(name="o_img_l", shape=[-1, 1] + OUT_IM_SIZE)
    label_a = fluid.data(name="label_a", shape=[-1, 1] + IM_SIZE, dtype="int64")
    label_b = fluid.data(name="label_b", shape=[-1, 1] + IM_SIZE, dtype="int64")
    w_a = fluid.data(name="w_a", shape=[-1] + IM_SIZE)
    w_b = fluid.data(name="w_b", shape=[-1] + IM_SIZE)
    with scope("signal_l"):
        encode_data, short_cuts = encode(img_l)
        decode_l = decode(encode_data, short_cuts)
        data = deconv(decode_l,
                      64,
                      filter_size=2,
                      stride=2,
                      padding=0)
        signal_l = double_conv(data, 1)
    with scope("signal_a"):
        encode_data, short_cuts = encode(img_l)
        decode_a = decode(encode_data, short_cuts)
        signal_a = get_logit(decode_a, SIGNAL_A_NUM)
    with scope("signal_b"):
        encode_data, short_cuts = encode(img_l)
        decode_b = decode(encode_data, short_cuts)
        signal_b = get_logit(decode_b, SIGNAL_B_NUM)

    cost_l = fluid.layers.square_error_cost(signal_l, o_img_l)
    loss_l = fluid.layers.mean(cost_l)

    cost_a_o = fluid.layers.softmax_with_cross_entropy(signal_a, label_a, axis=1)
    cost_b_o = fluid.layers.softmax_with_cross_entropy(signal_b, label_b, axis=1)
    cost = cost_a_o + cost_b_o
    loss = fluid.layers.mean(cost)

    test_program = train_program.clone(for_test=True)
    signal_a_out = fluid.layers.argmax(x=signal_a, axis=1)
    signal_b_out = fluid.layers.argmax(x=signal_b, axis=1)
    signal_sum = fluid.layers.concat([signal_a_out, signal_b_out], 1)

    cost_a = fluid.layers.elementwise_mul(cost_a_o, w_a)
    cost_b = fluid.layers.elementwise_mul(cost_b_o, w_b)
    loss_a = fluid.layers.mean(cost_a)
    loss_b = fluid.layers.mean(cost_b)

    learning_rate = fluid.layers.piecewise_decay(boundaries=BOUNDARIES, values=VALUES)
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
                                               WARM_UP_STEPS,
                                               START_LR,
                                               END_LR)
    opt = fluid.optimizer.Adam(decayed_lr)
    opt.minimize(loss_l)
    opt.minimize(loss_a)
    opt.minimize(loss_b)

train_reader = fluid.io.batch(
    reader=fluid.io.shuffle(reader(TRAIN_DATA_PATH, im_size=IM_SIZE), buf_size=4096),
    batch_size=BATCH_SIZE)
test_reader = fluid.io.batch(
    reader=reader(TEST_DATA_PATH, im_size=IM_SIZE),
    batch_size=BATCH_SIZE * 2)

feeder = fluid.DataFeeder(place=place, feed_list=["img_l", "o_img_l", "label_a", "label_b", "w_a", "w_b"],
                          program=train_program)

exe.run(start_program)
print("Net check --OK")
if os.path.exists(CHECK_POINT_DIR + ".pdopt") and LOAD_CHECKPOINT:
    fluid.io.load(train_program, CHECK_POINT_DIR, exe)


def if_exist(var):
    return os.path.exists(os.path.join(PER_MODEL_DIR, var.name))


if os.path.exists(PER_MODEL_DIR) and LOAD_PER_MODEL:
    fluid.io.load_vars(exe, PER_MODEL_DIR, train_program, predicate=if_exist)

MIN_LOSS = 10.
for epoch in range(EPOCH):
    out_loss_ab = list()
    out_loss_l = list()
    lr = None
    for data_id, data in enumerate(train_reader()):
        start_time = time.time()
        out = exe.run(program=train_program,
                      feed=feeder.feed(data),
                      fetch_list=[loss, loss_l, decayed_lr])
        out_loss_ab.append(out[0][0])
        out_loss_l.append(out[1][0])
        lr = out[2]
        cost_time = time.time() - start_time
        if data_id % (320 // BATCH_SIZE) == 0:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss_ab) / len(out_loss_ab)),
                  "L_Loss:{:.6f}".format(sum(out_loss_l) / len(out_loss_l)),
                  "\tTIME:\t{:.4f}/s".format(cost_time / BATCH_SIZE),
                  "\tLR:", lr)
            out_loss_ab = []
            out_loss_l = []
        if data_id % (3200 // BATCH_SIZE) == 1:
            out_loss_ab = []
            out_loss_l = []
            for data_t in test_reader():
                out = exe.run(program=test_program,
                              feed=feeder.feed(data_t),
                              fetch_list=[loss, loss_l])
                out_loss_ab.append(out[0][0])
                out_loss_l.append(out[1][0])
            test_loss = sum(out_loss_ab) / len(out_loss_ab)
            if test_loss <= MIN_LOSS:
                MIN_LOSS = test_loss
                fluid.io.save_inference_model(dirname=MODEL_DIR,
                                              feeded_var_names=["img_l"],
                                              target_vars=[signal_sum, signal_l],
                                              executor=exe,
                                              main_program=train_program)
            fluid.io.save(train_program, CHECK_POINT_DIR)
            print(epoch,
                  "TEST:\t{:.6f}".format(sum(out_loss_ab) / len(out_loss_ab)),
                  "L_Loss:{:.6f}".format(sum(out_loss_l) / len(out_loss_l)),
                  "\tMIN LOSS:\t{:.4f}".format(MIN_LOSS))
