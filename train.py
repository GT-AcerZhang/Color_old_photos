# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import numpy as np
import paddle.fluid as fluid

from l_net import l_net, set_name
from data_reader import reader, get_weight, get_class_num, get_resize

MODE = "AB"
LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False
DEBUG = False

ROOT_PATH = "./"
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, "data/train")
TEST_DATA_PATH = os.path.join(ROOT_PATH, "data/test")
CHECK_POINT_DIR = os.path.join(ROOT_PATH, "check_point/check_model.color")
PER_MODEL_DIR = os.path.join(ROOT_PATH, "data/unet_coco_v3")

EPOCH = 3 if not DEBUG else 1000
BATCH_SIZE = 8 if not DEBUG else 1
MAX_ITER_NUM = 100000

BOUNDARIES = [200, 500, 1000, 10000]
VALUES = [0.001, 0.00001, 0.000001, 0.0000005, 0.0000001]
WARM_UP_STEPS = 20
START_LR = 0.005
END_LR = 0.01

RESIZE = get_resize()
signal_ab_num = get_class_num()
CLASS_NUM = {"L": 1, "AB": signal_ab_num}
set_name(MODE)

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    # 训练输入层定义
    if MODE == "L":
        ipt_layer = fluid.data(name="ipt_layer_" + MODE, shape=[-1, 1, -1, -1])
        ipt_label = fluid.data(name="ipt_label_" + MODE, shape=[-1, 1, -1, -1])
    else:
        ipt_layer = fluid.data(name="ipt_layer_" + MODE, shape=[-1, 1, -1, -1])
        ipt_label = fluid.data(name="ipt_label_" + MODE, shape=[-1, 1, -1, -1], dtype="int64")
        w_ab = get_weight()
        ipt_w = fluid.layers.assign(w_ab)
        fluid.layers.Print(ipt_w)
        # ipt_w = fluid.layers.softmax(ipt_w)

    # 获得shape参数
    im_shape = fluid.layers.shape(ipt_label)
    im_shape = fluid.layers.slice(im_shape, axes=[0], starts=[2], ends=[4])

    # 组网
    signal = l_net(ipt_layer, im_shape, CLASS_NUM[MODE])

    if MODE == "L":
        # 统计指标
        loss = fluid.layers.mse_loss(signal, ipt_label)
        metric = loss
    elif MODE == "AB":
        signal2c = fluid.layers.reshape(signal, shape=[-1, CLASS_NUM[MODE], RESIZE * RESIZE])
        signal2c = fluid.layers.transpose(signal2c, [0, 2, 1])
        signal2acc = fluid.layers.flatten(signal2c, axis=2)

        label2c = fluid.layers.reshape(ipt_label, shape=[-1, 1, RESIZE * RESIZE])
        label2c = fluid.layers.transpose(label2c, [0, 2, 1])
        label2acc = fluid.layers.flatten(label2c, axis=2)
        metric = fluid.layers.accuracy(signal2acc, label2acc)
        metric_k3 = fluid.layers.accuracy(signal2acc, label2acc, 3)

        # 标签加权
        label2c = fluid.one_hot(label2c, CLASS_NUM[MODE])
        label2c = fluid.layers.squeeze(label2c, axes=[2])
        label2c = fluid.layers.elementwise_mul(label2c, ipt_w, axis=2)

        label2c.stop_gradient = True

        # 计算损失
        cost = fluid.layers.cross_entropy(fluid.layers.softmax(signal2c), label2c, soft_label=True)
        loss = fluid.layers.mean(cost)

    test_program = train_program.clone(for_test=True)

    # 学习率调度器
    learning_rate = fluid.layers.piecewise_decay(boundaries=BOUNDARIES, values=VALUES)
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
                                               WARM_UP_STEPS,
                                               START_LR,
                                               END_LR)
    clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
    opt = fluid.optimizer.Adam(decayed_lr,
                               regularization=fluid.regularizer.L2Decay(0.005),
                               grad_clip=clip)
    opt = fluid.optimizer.RecomputeOptimizer(opt)
    opt._set_checkpoints([ipt_layer, signal])
    opt.minimize(loss)

feeder_list = [ipt_layer, ipt_label]
train_feeder = fluid.DataFeeder(place=place,
                                feed_list=feeder_list,
                                program=train_program)
test_feeder = fluid.DataFeeder(place=place,
                               feed_list=feeder_list,
                               program=test_program)

if MODE == "L":
    train_loader = train_feeder.decorate_reader(reader(TRAIN_DATA_PATH, freeze_pix=MODE),
                                                multi_devices=True)
    test_loader = test_feeder.decorate_reader(reader(TEST_DATA_PATH, is_test=True, freeze_pix=MODE),
                                              multi_devices=True)
else:
    train_loader = train_feeder.decorate_reader(fluid.io.batch(reader(TRAIN_DATA_PATH, freeze_pix=MODE),
                                                               BATCH_SIZE,
                                                               drop_last=True),
                                                multi_devices=True)
    test_loader = test_feeder.decorate_reader(fluid.io.batch(reader(TEST_DATA_PATH, is_test=True, freeze_pix=MODE),
                                                             BATCH_SIZE,
                                                             drop_last=True),
                                              multi_devices=True)

exe.run(start_program)

compiled_train_prog = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=loss.name)
compiled_test_prog = fluid.CompiledProgram(test_program).with_data_parallel(share_vars_from=compiled_train_prog)
print("Net check --OK")
if os.path.exists(CHECK_POINT_DIR) and LOAD_CHECKPOINT:
    fluid.io.load_persistables(exe, CHECK_POINT_DIR, train_program)


def if_exist(var):
    return os.path.exists(os.path.join(PER_MODEL_DIR, var.name))


if os.path.exists(PER_MODEL_DIR) and LOAD_PER_MODEL:
    fluid.io.load_vars(exe, PER_MODEL_DIR, train_program, predicate=if_exist)

BEST_METRIC = 5.
ITER_NUM = 0
for epoch in range(EPOCH):
    out_loss = list()
    out_metric = list()
    lr = None

    for data_id, data in enumerate(train_loader()):
        if data_id == 0:
            print("\033[0;37;42mEpoch", epoch, "data load done\033[0m")
        ITER_NUM += 1
        if ITER_NUM == MAX_ITER_NUM:
            print("\033[0;37;41m程序已成功到达指定iter数量\033[0m")
            break
        start_time = time.time()
        out = exe.run(program=compiled_train_prog,
                      feed=data,
                      fetch_list=[loss, metric, decayed_lr])
        out_loss.append(out[0][0])
        out_metric.append(out[1][0])
        lr = out[2]
        cost_time = time.time() - start_time
        if DEBUG:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss) / len(out_loss)),
                  "\tMETRIC([L]PSNR | [AB]ACC):{:.6f}".format(
                      10 * np.log10(255 * 255 / (sum(out_metric) / len(out_metric))) if MODE == "L"
                      else np.average(out_metric)),
                  "\tTIME:\t{:.4f}/s".format(cost_time / len(data)),
                  "\tLR:", lr)
            if epoch % 50 == 50 - 1:
                fluid.io.save_persistables(exe, "DEBUG_" + CHECK_POINT_DIR, train_program)
            break
        # if data_id == epoch:
        #     print(data_id, "break")
        #     break
        if data_id % 50 == 0:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss) / len(out_loss)),
                  "\tMETRIC([L]PSNR | [AB]ACC):{:.6f}".format(
                      10 * np.log10(255 * 255 / (sum(out_metric) / len(out_metric))) if MODE == "L"
                      else np.average(out_metric)),
                  "\tTIME:\t{:.4f}/s".format(cost_time / len(data)),
                  "\tLR:", lr)
            out_loss = []
            out_loss_l = []
            print("\033[0;37;41m[WARNING]\tSaving checkpoint... Please don't stop running! \033[0m")
            fluid.io.save_persistables(exe, CHECK_POINT_DIR, train_program)
            print("\033[0;37;42m[INFO]\tDone\033[0m")

        if data_id % 300 == 300 - 1:
            out_loss = list()
            out_metric = list()
            out_metric_k3 = list()
            print("Run test...")
            for t_data_id, data_t in enumerate(test_loader()):
                if t_data_id == 160 // BATCH_SIZE:
                    break
                fetch_list = [loss, metric] if MODE == "L" else [loss, metric, metric_k3]
                out = exe.run(program=compiled_test_prog,
                              feed=data_t,
                              fetch_list=fetch_list)
                out_loss.append(out[0][0])
                out_metric.append(out[1][0])
                if MODE != "L":
                    out_metric_k3.append(out[2][0])
            avg_metric = np.average(out_metric)
            avg_metric_k3 = np.average(out_metric_k3) if MODE != "L" else 0
            avg_loss = np.average(out_loss)
            if avg_loss <= BEST_METRIC:
                BEST_METRIC = avg_loss
                print("\033[0;37;41m[WARNING]\tSaving best checkpoint... Please don't stop running! \033[0m")
                fluid.io.save_persistables(exe, CHECK_POINT_DIR + ".best", train_program)
                print("\033[0;37;42m[INFO]\tDone\033[0m")
            print(epoch,
                  "-",
                  data_id,
                  "TEST:\t{:.6f}".format(sum(out_loss) / len(out_loss)),
                  "\tMETRIC([L]PSNR | [AB]ACC):{:.6f}".format(
                      10 * np.log10(255 * 255 / (sum(out_metric) / len(out_metric))) if MODE == "L"
                      else avg_metric),
                  "\tBEST METRIC", BEST_METRIC)
            if MODE != "L":
                print("ACC Top-K", avg_metric_k3)
