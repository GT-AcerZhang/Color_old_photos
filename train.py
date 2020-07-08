# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import paddle.fluid as fluid

from models.modeling.unet import unet
from models.modeling.hrnet import hrnet
from data_reader import reader


LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False

TRAIN_DATA_PATH = "./data/train"
TEST_DATA_PATH = "./data/test"
CHECK_POINT_DIR = "./check_model.color"
PER_MODEL_DIR = "./data/unet_coco_v3"
MODEL_DIR = "./best_model.color"

EPOCH = 5
BATCH_SIZE = 8
IM_SIZE = [512] * 2
BOUNDARIES = [1000, 2000, 5000, 10000]
VALUES = [0.01, 0.005, 0.001, 0.0005, 0.0001]
WARM_UP_STEPS = 150
START_LR = 0.005
END_LR = 0.01

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    inp_img = fluid.data(name="inp_img", shape=[-1, 1] + IM_SIZE)
    ori_img = fluid.data(name="ori_img", shape=[-1, 2] + IM_SIZE)
    out_put = unet(inp_img, 2)
    # out_put = hrnet(inp_img, 2)
    out_put_c = fluid.layers.flatten(out_put)
    ori_img = fluid.layers.flatten(ori_img)
    cost = fluid.layers.square_error_cost(out_put_c, ori_img)
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
    reader=fluid.io.shuffle(reader(TRAIN_DATA_PATH, IM_SIZE), buf_size=4096),
    batch_size=BATCH_SIZE)
test_reader = fluid.io.batch(
    reader=reader(TEST_DATA_PATH, IM_SIZE),
    batch_size=BATCH_SIZE * 2)

feeder = fluid.DataFeeder(place=place, feed_list=["inp_img", "ori_img"], program=train_program)

exe.run(start_program)
if os.path.exists(CHECK_POINT_DIR + ".pdopt") and LOAD_CHECKPOINT:
    fluid.io.load(train_program, CHECK_POINT_DIR, exe)


def if_exist(var):
    return os.path.exists(os.path.join(PER_MODEL_DIR, var.name))


if os.path.exists(PER_MODEL_DIR) and LOAD_PER_MODEL:
    fluid.io.load_vars(exe, PER_MODEL_DIR, train_program, predicate=if_exist)

MIN_LOSS = 1.
for epoch in range(EPOCH):
    start_time = time.time()
    out_loss = list()
    lr = None
    for data_id, data in enumerate(train_reader()):
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
            start_time = time.time()
            out_loss = list()
        if data_id % 100 == 0:
            out_loss = list()
            for data in test_reader():
                out = exe.run(program=test_program,
                              feed=feeder.feed(data),
                              fetch_list=[loss])
                out_loss.append(out[0][0])
            test_loss = sum(out_loss) / len(out_loss)

            if test_loss <= MIN_LOSS:
                MIN_LOSS = test_loss
                fluid.io.save_inference_model(dirname=MODEL_DIR,
                                              feeded_var_names=["inp_img"],
                                              target_vars=[out_put],
                                              executor=exe,
                                              main_program=train_program)
            fluid.io.save(train_program, CHECK_POINT_DIR)
            print(epoch, "TEST :\t{:.6f}".format(test_loss), "\tMIN LOSS:\t{:.6f}".format(MIN_LOSS))
