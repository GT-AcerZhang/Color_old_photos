# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import numpy as np
import paddle.fluid as fluid

from models.libs.model_libs import scope
from models.modeling.l_net import l_net
from data_reader import reader, get_weight, get_class_num, get_resize

MODE = "A"
LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False

ROOT_PATH = "./"
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, "data/train")
TEST_DATA_PATH = os.path.join(ROOT_PATH, "data/test")
CHECK_POINT_DIR = os.path.join(ROOT_PATH, "check_point/check_model.color")
PER_MODEL_DIR = os.path.join(ROOT_PATH, "data/unet_coco_v3")
MODEL_DIR = os.path.join(ROOT_PATH, "best_model.color")

EPOCH = 5
BATCH_SIZE = 16
MAX_ITER_NUM = 100000

BOUNDARIES = [500, 1000, 5000, 10000]
VALUES = [0.001, 0.00005, 0.00001, 0.000005, 0.000001]
WARM_UP_STEPS = 50
START_LR = 0.005
END_LR = 0.01

RESIZE = get_resize()
signal_a_num, signal_b_num = get_class_num()
CLASS_NUM = {"L": 1, "A": signal_a_num, "B": signal_b_num}

place = fluid.CUDAPlace(0)
places = fluid.cuda_places()
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
        w_a, w_b = get_weight()
        ipt_w = fluid.layers.assign(w_a if MODE == "A" else w_b)

    # 获得shape参数
    im_shape = fluid.layers.shape(ipt_label)
    im_shape = fluid.layers.slice(im_shape, axes=[0], starts=[2], ends=[4])

    # 组网
    with scope("signal_" + MODE):
        signal = l_net(ipt_layer, im_shape, CLASS_NUM[MODE])

    if MODE == "L":
        # 统计指标
        loss = fluid.layers.mse_loss(signal, ipt_label)
        metric = loss
    else:
        signal2c = fluid.layers.reshape(signal, shape=[-1, CLASS_NUM[MODE], RESIZE * RESIZE])
        signal2c = fluid.layers.transpose(signal2c, [0, 2, 1])
        signal2acc = fluid.layers.flatten(signal2c, axis=2)

        label2c = fluid.layers.reshape(ipt_label, shape=[-1, 1, RESIZE * RESIZE])
        label2c = fluid.layers.transpose(label2c, [0, 2, 1])
        label2acc = fluid.layers.flatten(label2c, axis=2)
        metric = fluid.layers.accuracy(signal2acc, label2acc)

        # 标签加权
        label2c = fluid.one_hot(label2c, CLASS_NUM[MODE])
        label2c = fluid.layers.squeeze(label2c, axes=[2])
        label2c = fluid.layers.elementwise_mul(label2c, ipt_w, axis=2)

        # 计算损失
        cost = fluid.layers.cross_entropy(fluid.layers.softmax(signal2c), label2c, soft_label=True)
        loss = fluid.layers.mean(cost)

    test_program = train_program.clone(for_test=True)
    # signal_a = fluid.layers.resize_nearest(signal_a, im_shape)
    # signal_b = fluid.layers.resize_nearest(signal_b, im_shape)
    # signal_a_out = fluid.layers.transpose(signal_a, [0, 2, 3, 1])
    # signal_b_out = fluid.layers.transpose(signal_b, [0, 2, 3, 1])
    # signal_a_out = fluid.layers.argmax(x=signal_a_out, axis=-1)
    # signal_b_out = fluid.layers.argmax(x=signal_b_out, axis=-1)

    # 学习率调度器
    learning_rate = fluid.layers.piecewise_decay(boundaries=BOUNDARIES, values=VALUES)
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
                                               WARM_UP_STEPS,
                                               START_LR,
                                               END_LR)
    # opt = fluid.optimizer.Adam(decayed_lr, grad_clip=fluid.clip.GradientClipByNorm(clip_norm=1.0))
    opt = fluid.optimizer.Adam(decayed_lr)
    opt.minimize(loss)

feeder_list = [ipt_layer, ipt_label]
train_feeder = fluid.DataFeeder(place=place,
                                feed_list=feeder_list,
                                program=train_program)
test_feeder = fluid.DataFeeder(place=place,
                               feed_list=feeder_list,
                               program=test_program)

train_loader = train_feeder.decorate_reader(fluid.io.batch(reader(TRAIN_DATA_PATH, freeze_pix=MODE), BATCH_SIZE),
                                            multi_devices=True)
test_loader = test_feeder.decorate_reader(fluid.io.batch(reader(TEST_DATA_PATH, is_test=True, freeze_pix=MODE),
                                                         BATCH_SIZE),
                                          multi_devices=True)

exe.run(start_program)

compiled_train_prog = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=loss.name)
compiled_test_prog = fluid.CompiledProgram(test_program).with_data_parallel(share_vars_from=compiled_train_prog)
print("Net check --OK")
if os.path.exists(CHECK_POINT_DIR + ".pdopt") and LOAD_CHECKPOINT:
    fluid.io.load(train_program, CHECK_POINT_DIR, exe)


def if_exist(var):
    return os.path.exists(os.path.join(PER_MODEL_DIR, var.name))


if os.path.exists(PER_MODEL_DIR) and LOAD_PER_MODEL:
    fluid.io.load_vars(exe, PER_MODEL_DIR, train_program, predicate=if_exist)

BEST_METRIC = 0.
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

        if data_id % 20 == 0:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss) / len(out_loss)),
                  "\tMETRIC([L]PSNR|[AB]ACC):{:.6f}".format(
                      10 * np.log10(255 * 255 / sum(out_metric) / len(out_metric)) if MODE == "L"
                      else np.average(out_metric)),
                  "\tTIME:\t{:.4f}/s".format(cost_time / len(data)),
                  "\tLR:", lr)
            out_loss = []
            out_loss_l = []
            print("\033[0;37;41m[WARNING]\tSaving checkpoint... Please don't stop running! \033[0m")
            fluid.io.save(train_program, CHECK_POINT_DIR)
            print("\033[0;37;42m[INFO]\tDone\033[0m")

        if data_id % 100 == 100 - 1:
            out_loss = list()
            out_metric = list()
            print("Run test...")
            for t_data_id, data_t in enumerate(test_loader()):
                if t_data_id == 160 // BATCH_SIZE:
                    break
                out = exe.run(program=compiled_test_prog,
                              feed=data_t,
                              fetch_list=[loss, metric])
                out_loss.append(out[0][0])
                out_metric.append(out[1][0])
            avg_metric = np.average(out_metric)
            if (avg_metric <= BEST_METRIC and MODE == "L") or (avg_metric >= BEST_METRIC and MODE != "L"):
                BEST_METRIC = avg_metric
                print("\033[0;37;41m[WARNING]\tSaving best checkpoint... Please don't stop running! \033[0m")
                fluid.io.save(train_program, CHECK_POINT_DIR + ".best")
                print("\033[0;37;42m[INFO]\tDone\033[0m")
            print(epoch,
                  "-",
                  data_id,
                  "TEST:\t{:.6f}".format(sum(out_loss) / len(out_loss)),
                  "\tMETRIC([L]PSNR | [AB]ACC):{:.6f}".format(
                      10 * np.log10(255 * 255 / sum(out_metric) / len(out_metric)) if MODE == "L"
                      else avg_metric),
                  "\tBEST METRIC", BEST_METRIC)
