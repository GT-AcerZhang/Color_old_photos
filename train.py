# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import paddle.fluid as fluid

from models.libs.model_libs import scope
from models.modeling.unet import unet
from data_reader import reader

LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False
GPU_NUM = 1

TRAIN_DATA_PATH = "./data/train"
TEST_DATA_PATH = "./data/test"
CHECK_POINT_DIR = "./check_point/check_model.color"
PER_MODEL_DIR = "./data/unet_coco_v3"
MODEL_DIR = "./best_model.color"

EPOCH = 100
BATCH_SIZE = 8
IM_SIZE = [256] * 2
OUT_IM_SIZE = [IM_SIZE[0] * 2, IM_SIZE[1] * 2]
SIGNAL_A_NUM = 21
SIGNAL_B_NUM = 23

BOUNDARIES = [1000, 5000, 15000, 100000]
VALUES = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
WARM_UP_STEPS = 150
START_LR = 0.005
END_LR = 0.01

parallel_places = [fluid.CUDAPlace(i) for i in range(GPU_NUM)]
# place = fluid.CUDAPlace(0)
place = parallel_places[0]
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    img_l = fluid.data(name="img_l", shape=[-1, 1] + IM_SIZE)
    o_img_l = fluid.data(name="o_img_l", shape=[-1, 1] + IM_SIZE)
    label_a = fluid.data(name="label_a", shape=[-1, 1] + IM_SIZE, dtype="int64")
    label_b = fluid.data(name="label_b", shape=[-1, 1] + IM_SIZE, dtype="int64")
    w_a = fluid.data(name="w_a", shape=[-1] + IM_SIZE)
    w_b = fluid.data(name="w_b", shape=[-1] + IM_SIZE)

    with scope("signal_l"):
        signal_l = unet(img_l, 1)
    with scope("signal_a"):
        signal_a = unet(img_l, SIGNAL_A_NUM)
    with scope("signal_b"):
        signal_b = unet(img_l, SIGNAL_B_NUM)

    loss_l = fluid.layers.mse_loss(signal_l, o_img_l)
    # loss_l = fluid.layers.mean(cost_l)

    cost_a_o = fluid.layers.softmax_with_cross_entropy(signal_a, label_a, axis=1)
    cost_b_o = fluid.layers.softmax_with_cross_entropy(signal_b, label_b, axis=1)
    cost = cost_a_o + cost_b_o
    loss = fluid.layers.mean(cost)

    test_program = train_program.clone(for_test=True)
    signal_a_out = fluid.layers.argmax(x=signal_a, axis=1)
    signal_b_out = fluid.layers.argmax(x=signal_b, axis=1)

    # cost_a = fluid.layers.elementwise_mul(cost_a_o, w_a)
    # cost_b = fluid.layers.elementwise_mul(cost_b_o, w_b)
    # loss_a = fluid.layers.mean(cost_a)
    # loss_b = fluid.layers.mean(cost_b)

    learning_rate = fluid.layers.piecewise_decay(boundaries=BOUNDARIES, values=VALUES)
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
                                               WARM_UP_STEPS,
                                               START_LR,
                                               END_LR)
    opt = fluid.optimizer.Adam(decayed_lr)
    opt.minimize(loss_l)
    opt.minimize(loss)
    final_loss = loss_l + loss

train_loader = fluid.io.DataLoader.from_generator(feed_list=[img_l, o_img_l, label_a, label_b, w_a, w_b],
                                                  capacity=16,
                                                  iterable=True,
                                                  use_double_buffer=True,
                                                  drop_last=True)
train_loader.set_sample_generator(reader(TRAIN_DATA_PATH, im_size=IM_SIZE), BATCH_SIZE, drop_last=True, places=place)
test_loader = fluid.io.DataLoader.from_generator(feed_list=[img_l, o_img_l, label_a, label_b, w_a, w_b],
                                                 capacity=8,
                                                 iterable=True,
                                                 use_double_buffer=True,
                                                 drop_last=True)
test_loader.set_sample_generator(reader(TEST_DATA_PATH, im_size=IM_SIZE), BATCH_SIZE, drop_last=True, places=place)

exe.run(start_program)

compiled_train_prog = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=final_loss.name,
                                                                              places=parallel_places)
compiled_test_prog = fluid.CompiledProgram(test_program).with_data_parallel(share_vars_from=compiled_train_prog,
                                                                            places=parallel_places)
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
    for data_id, data in enumerate(train_loader()):
        start_time = time.time()
        out = exe.run(program=compiled_train_prog,
                      feed=data,
                      fetch_list=[loss, loss_l, decayed_lr])
        out_loss_ab.append(out[0][0])
        out_loss_l.append(out[1][0])
        lr = out[2]
        cost_time = time.time() - start_time
        if data_id % (320 // BATCH_SIZE) == (320 // BATCH_SIZE) - 1:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss_ab) / len(out_loss_ab)),
                  "L_Loss:{:.6f}".format(sum(out_loss_l) / len(out_loss_l)),
                  "\tTIME:\t{:.4f}/s".format(cost_time / BATCH_SIZE),
                  "\tLR:", lr)
            out_loss_ab = []
            out_loss_l = []
        fluid.io.save(train_program, CHECK_POINT_DIR)
        if data_id % (6400 // BATCH_SIZE) == (6400 // BATCH_SIZE) - 1:
            out_loss_ab = []
            out_loss_l = []
            for data_t in test_loader:
                out = exe.run(program=compiled_test_prog,
                              feed=data_t,
                              fetch_list=[loss, loss_l])
                out_loss_ab.append(out[0][0])
                out_loss_l.append(out[1][0])
            test_loss = sum(out_loss_ab) / len(out_loss_ab)
            if test_loss <= MIN_LOSS:
                MIN_LOSS = test_loss
                fluid.io.save_inference_model(dirname=MODEL_DIR,
                                              feeded_var_names=["img_l"],
                                              target_vars=[signal_l, signal_a_out, signal_b_out],
                                              executor=exe,
                                              main_program=train_program)

            print(epoch,
                  "TEST:\t{:.6f}".format(sum(out_loss_ab) / len(out_loss_ab)),
                  "L_Loss:{:.8f}".format(sum(out_loss_l) / len(out_loss_l)),
                  "\tMIN LOSS:\t{:.4f}".format(MIN_LOSS))
