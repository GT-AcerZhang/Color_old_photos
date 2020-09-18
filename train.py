# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

from tqdm import tqdm
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import profiler

from color_net import l_net, ab_net, set_name
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
BATCH_SIZE = 16 if not DEBUG else 8
MAX_ITER_NUM = 100000

BOUNDARIES = [50000, 100000, 500000, 1000000]
VALUES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
WARM_UP_STEPS = int(1000 / BATCH_SIZE) if not DEBUG else 10
START_LR = 0.01
END_LR = 0.05

GAMMA_A = 1.
GAMMA_B = 1.

RESIZE = get_resize()
signal_a_num, signal_b_num = get_class_num()
CLASS_NUM = {"L": 1, "A": signal_a_num, "B": signal_b_num}

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    # 设置参数标识符
    set_name(MODE)
    # 训练输入层定义
    if MODE == "L":
        ipt_layer = fluid.data(name="ipt_layer_" + MODE, shape=[-1, 1, -1, -1])
        ipt_label = fluid.data(name="ipt_label_" + MODE, shape=[-1, 1, -1, -1])
    else:
        ipt_layer = fluid.data(name="ipt_layer_" + MODE, shape=[-1, 1, -1, -1])
        ipt_label_a = fluid.data(name="ipt_label_a", shape=[-1, 1, -1, -1], dtype="int64")
        ipt_label_b = fluid.data(name="ipt_label_b", shape=[-1, 1, -1, -1], dtype="int64")
        w_a, w_b = get_weight()
        ipt_wa = fluid.layers.assign(w_a)
        ipt_wb = fluid.layers.assign(w_b)
        # ipt_w = fluid.layers.softmax(ipt_w)

    # 获得shape参数
    im_shape = fluid.layers.shape(ipt_label_a)
    im_shape = fluid.layers.slice(im_shape, axes=[0], starts=[2], ends=[4])

    if MODE == "L":
        # 组网
        signal = l_net(ipt_layer, im_shape, CLASS_NUM[MODE])
        # 统计指标
        loss = fluid.layers.mse_loss(signal, ipt_label)
        metric = loss
    elif MODE == "AB":
        # 组网
        a_global, b_global, a_classify, b_classify, root_signal = ab_net(ipt_layer,
                                                                         im_shape,
                                                                         CLASS_NUM["A"],
                                                                         CLASS_NUM["B"])
        # 分类部分
        a_signal2c = fluid.layers.reshape(a_classify, shape=[0, CLASS_NUM["A"], RESIZE * RESIZE])
        a_signal2c = fluid.layers.transpose(a_signal2c, [0, 2, 1])
        b_signal2c = fluid.layers.reshape(b_classify, shape=[0, CLASS_NUM["B"], RESIZE * RESIZE])
        b_signal2c = fluid.layers.transpose(b_signal2c, [0, 2, 1])

        # 准确率统计指标
        a_signal2acc = fluid.layers.flatten(a_signal2c, axis=2)
        b_signal2acc = fluid.layers.flatten(b_signal2c, axis=2)

        a_label2c = fluid.layers.reshape(ipt_label_a, shape=[0, 1, RESIZE * RESIZE])
        a_label2c = fluid.layers.transpose(a_label2c, [0, 2, 1])
        a_label2acc = fluid.layers.flatten(a_label2c, axis=2)
        b_label2c = fluid.layers.reshape(ipt_label_b, shape=[0, 1, RESIZE * RESIZE])
        b_label2c = fluid.layers.transpose(b_label2c, [0, 2, 1])
        b_label2acc = fluid.layers.flatten(b_label2c, axis=2)

        metric = (fluid.layers.accuracy(a_signal2acc, a_label2acc) +
                  fluid.layers.accuracy(b_signal2acc, b_label2acc)) * 0.5
        metric_k3 = (fluid.layers.accuracy(a_signal2acc, a_label2acc, 3) +
                     fluid.layers.accuracy(b_signal2acc, b_label2acc, 3)) * 0.5

        # 标签加权
        a_label2c = fluid.one_hot(a_label2c, CLASS_NUM["A"])
        a_label2c = fluid.layers.squeeze(a_label2c, axes=[2])
        a_label2c = fluid.layers.elementwise_mul(a_label2c, ipt_wa, axis=2)

        b_label2c = fluid.one_hot(b_label2c, CLASS_NUM["B"])
        b_label2c = fluid.layers.squeeze(b_label2c, axes=[2])
        b_label2c = fluid.layers.elementwise_mul(b_label2c, ipt_wb, axis=2)

        # 计算损失
        a_signal2c = fluid.layers.softmax(a_signal2c)
        b_signal2c = fluid.layers.softmax(b_signal2c)

        cost_a = fluid.layers.cross_entropy(a_signal2c, a_label2c, soft_label=True)
        cost_b = fluid.layers.cross_entropy(b_signal2c, b_label2c, soft_label=True)
        # max_a = fluid.layers.reduce_max(a_signal2c, dim=-1, keep_dim=True)
        # max_b = fluid.layers.reduce_max(b_signal2c, dim=-1, keep_dim=True)
        # alpha_a = fluid.layers.ones_like(max_a) - max_a
        # alpha_b = fluid.layers.ones_like(max_a) - max_b
        # cost_a *= alpha_a * GAMMA_A
        # cost_b *= alpha_b * GAMMA_B
        loss = fluid.layers.mean(cost_a + cost_b)

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
    opt._set_checkpoints(root_signal)
    opt.minimize(loss)

feeder_list = [ipt_layer, ipt_label] if MODE == "L" else [ipt_layer, ipt_label_a, ipt_label_b]
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
        print("\033[0;37;42mIter", data_id, "mini batch load done\033[0m", end="\r")
        if data_id == 0 and epoch == 0:
            pass
            # profiler.start_profiler("All")
        ITER_NUM += 1
        if ITER_NUM == MAX_ITER_NUM:
            print("\033[0;37;41m程序已成功到达指定iter数量\033[0m")
            break
        start_time = time.time()
        out = exe.run(program=compiled_train_prog,
                      feed=data,
                      fetch_list=[loss, metric, decayed_lr])
        if data_id == 0 and epoch == 0:
            # profiler.stop_profiler("total", "/profile")
            pass
        out_loss.append(out[0][0])
        out_metric.append(out[1][0])
        lr = out[2]
        cost_time = time.time() - start_time
        if DEBUG and data_id == BATCH_SIZE - 1:
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
