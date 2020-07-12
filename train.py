# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import paddle.fluid as fluid

from models.modeling.unet import unet
from data_reader import reader

LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False

TRAIN_DATA_PATH = "./data/train"
TEST_DATA_PATH = "./data/test"
CHECK_POINT_DIR = "./check_model.color"
PER_MODEL_DIR = "./data/unet_coco_v3"
MODEL_DIR = "./best_model.color"

MULTI = False
EPOCH = 500
BATCH_SIZE = 8
IM_SIZE = [256] * 2
COLOR_NUM = 300

BOUNDARIES = [1000, 2000, 5000, 10000]
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
    img = fluid.data(name="img", shape=[-1, 1] + IM_SIZE)
    label = fluid.data(name="label", shape=[-1, 1] + IM_SIZE, dtype="float32")

    signal = unet(img, 1)
    signal_out = signal
    cost = fluid.layers.square_error_cost(signal_out, label)
    # signal_out = fluid.layers.argmax(x=signal, axis=1)
    # label = fluid.layers.flatten(label)
    # label = fluid.layers.reshape(label, (-1, IM_SIZE[0] * IM_SIZE[1], 1))
    # signal = fluid.layers.reshape(signal, (-1, IM_SIZE[0] * IM_SIZE[1], COLOR_NUM))
    # label = fluid.one_hot(label, COLOR_NUM)
    # label = fluid.layers.squeeze(label, axes=[2])
    #
    # cost = fluid.layers.sigmoid_cross_entropy_with_logits(signal, label)
    loss = fluid.layers.mean(cost)
    test_program = train_program.clone(for_test=True)

    learning_rate = fluid.layers.piecewise_decay(boundaries=BOUNDARIES, values=VALUES)
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
                                               WARM_UP_STEPS,
                                               START_LR,
                                               END_LR)
    opt = fluid.optimizer.Adamax(decayed_lr)
    opt.minimize(loss)

train_reader = fluid.io.batch(
    reader=fluid.io.shuffle(reader(TRAIN_DATA_PATH), buf_size=4096),
    batch_size=BATCH_SIZE)
test_reader = fluid.io.batch(
    reader=reader(TEST_DATA_PATH),
    batch_size=BATCH_SIZE * 2)

feeder = fluid.DataFeeder(place=place, feed_list=["img", "label"], program=train_program)

exe.run(start_program)
if os.path.exists(CHECK_POINT_DIR + ".pdopt") and LOAD_CHECKPOINT:
    fluid.io.load(train_program, CHECK_POINT_DIR, exe)


def if_exist(var):
    return os.path.exists(os.path.join(PER_MODEL_DIR, var.name))


if os.path.exists(PER_MODEL_DIR) and LOAD_PER_MODEL:
    fluid.io.load_vars(exe, PER_MODEL_DIR, train_program, predicate=if_exist)

if MULTI is True:
    train_program = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=loss.name)
    test_exe = fluid.Executor(place)
else:
    test_exe = exe

MIN_LOSS = 10.
for epoch in range(EPOCH):
    out_loss = list()
    lr = None
    for data_id, data in enumerate(train_reader()):
        start_time = time.time()
        out = exe.run(program=train_program,
                      feed=feeder.feed(data),
                      fetch_list=[loss, decayed_lr])
        out_loss.append(out[0][0])
        lr = out[1]
        cost_time = time.time() - start_time
        if data_id % 10 == 0:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss) / len(out_loss)),
                  "\tTIME:\t{:.4f}/s".format(cost_time / BATCH_SIZE),
                  "\tLR:", lr)
            out_loss = list()
        if data_id % 100 == 0:
            out_loss = list()
            for data in test_reader():
                out = test_exe.run(program=test_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[loss])
                out_loss.append(out[0][0])
            test_loss = sum(out_loss) / len(out_loss)
            out_loss = list()
            if test_loss <= MIN_LOSS:
                MIN_LOSS = test_loss
                fluid.io.save_inference_model(dirname=MODEL_DIR,
                                              feeded_var_names=["img"],
                                              target_vars=[signal_out],
                                              executor=exe,
                                              main_program=train_program)
            fluid.io.save(train_program, CHECK_POINT_DIR)
            print(epoch, "TEST :\t{:.6f}".format(test_loss), "\tMIN LOSS:\t{:.6f}".format(MIN_LOSS))
