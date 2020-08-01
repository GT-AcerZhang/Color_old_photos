# Author: Acer Zhang
# Datetime:2020/7/6 19:37
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os
import time

import numpy as np
import paddle.fluid as fluid

from models.libs.model_libs import scope
from models.modeling.l_net import l_net, encode, decode
from data_reader import reader, get_weight, get_class_num

LOAD_CHECKPOINT = False
LOAD_PER_MODEL = False
FREEZE_PIX = True

ROOT_PATH = "./"
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, "data/train")
TEST_DATA_PATH = os.path.join(ROOT_PATH, "data/test")
CHECK_POINT_DIR = os.path.join(ROOT_PATH, "check_point/check_model.color")
PER_MODEL_DIR = os.path.join(ROOT_PATH, "data/unet_coco_v3")
MODEL_DIR = os.path.join(ROOT_PATH, "best_model.color")

EPOCH = 1
BATCH_SIZE = 32
MAX_ITER_NUM = 100000

BOUNDARIES = [10000, 15000, 50000, 100000]
VALUES = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
WARM_UP_STEPS = 500
START_LR = 0.005
END_LR = 0.01

SIGNAL_A_NUM, SIGNAL_B_NUM = get_class_num()

place = fluid.CUDAPlace(0)
places = fluid.cuda_places()
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    # 训练输入层定义
    # 缩放后数据以及原始数据的灰度图像
    resize_l = fluid.data(name="resize_l", shape=[-1, 1, -1, -1])
    img_l = fluid.data(name="img_l", shape=[-1, 1, -1, -1])

    # LAB三通道
    img_l2 = fluid.data(name="img_l2", shape=[-1, 1, -1, -1])
    label_a = fluid.data(name="label_a", shape=[-1, 1, -1, -1], dtype="int64")
    label_b = fluid.data(name="label_b", shape=[-1, 1, -1, -1], dtype="int64")
    w_a, w_b = get_weight()
    w_a = fluid.layers.assign(w_a)
    w_b = fluid.layers.assign(w_b)

    # 获得shape参数
    im_shape = fluid.layers.shape(img_l2 if FREEZE_PIX else resize_l)
    im_shape = fluid.layers.slice(im_shape, axes=[0], starts=[2], ends=[4])
    im_shape2 = fluid.layers.shape(img_l2)
    im_shape2 = fluid.layers.slice(im_shape2, axes=[0], starts=[2], ends=[4])

    # 组网
    with scope("signal_a"):
        encode_data_a, short_cuts_a = encode(img_l2)
        decode_data_a = decode(encode_data_a, short_cuts_a, im_shape2)
        signal_a = fluid.layers.conv2d(decode_data_a, SIGNAL_A_NUM, 1, 1, act="softmax")
    with scope("signal_b"):
        encode_data_b, short_cuts_b = encode(img_l2)
        decode_data_b = decode(encode_data_b, short_cuts_b, im_shape2)
        signal_b = fluid.layers.conv2d(decode_data_b, SIGNAL_B_NUM, 1, 1, act="softmax")
    if not FREEZE_PIX:
        with scope("signal_l"):
            signal_l = l_net(resize_l, im_shape, 1)
            loss_l = fluid.layers.mse_loss(signal_l, img_l)
    else:
        signal_l = signal_a

    # 计算AB损失和统计指标准备工作
    # 转换网络输出
    signal_a2c = fluid.layers.transpose(signal_a, [0, 2, 3, 1])
    signal_b2c = fluid.layers.transpose(signal_b, [0, 2, 3, 1])
    signal_a2c = fluid.layers.reshape(signal_a2c, shape=[-1, 256 * 256, SIGNAL_A_NUM])
    signal_b2c = fluid.layers.reshape(signal_b2c, shape=[-1, 256 * 256, SIGNAL_B_NUM])
    signal_a_acc = fluid.layers.reshape(signal_a2c, shape=[-1, SIGNAL_A_NUM])
    signal_b_acc = fluid.layers.reshape(signal_b2c, shape=[-1, SIGNAL_B_NUM])

    # 转化label和权重
    label_a2c = fluid.layers.transpose(label_a, [0, 2, 3, 1])
    label_b2c = fluid.layers.transpose(label_b, [0, 2, 3, 1])
    label_a2c = fluid.layers.reshape(label_a2c, shape=[-1, 256 * 256, 1])
    label_b2c = fluid.layers.reshape(label_b2c, shape=[-1, 256 * 256, 1])
    label_a_acc = fluid.layers.reshape(label_a2c, shape=[-1, 1])
    label_b_acc = fluid.layers.reshape(label_b2c, shape=[-1, 1])

    # 统计指标
    a_acc = fluid.layers.accuracy(signal_a_acc, label_a_acc)
    b_acc = fluid.layers.accuracy(signal_b_acc, label_b_acc)

    label_a2c = fluid.one_hot(label_a2c, SIGNAL_A_NUM)
    label_b2c = fluid.one_hot(label_b2c, SIGNAL_B_NUM)
    label_a2c = fluid.layers.squeeze(label_a2c, axes=[2])
    label_b2c = fluid.layers.squeeze(label_b2c, axes=[2])

    # 对label进行加权
    label_a2c = fluid.layers.elementwise_mul(label_a2c, w_a, axis=2)
    label_b2c = fluid.layers.elementwise_mul(label_b2c, w_b, axis=2)

    cost_a = fluid.layers.cross_entropy(signal_a2c, label_a2c, soft_label=True)
    cost_b = fluid.layers.cross_entropy(signal_b2c, label_b2c, soft_label=True)
    cost_ab = cost_a + cost_b
    loss_ab = fluid.layers.mean(cost_ab)

    test_program = train_program.clone(for_test=True)
    signal_a = fluid.layers.resize_nearest(signal_a, im_shape)
    signal_b = fluid.layers.resize_nearest(signal_b, im_shape)
    signal_a_out = fluid.layers.transpose(signal_a, [0, 2, 3, 1])
    signal_b_out = fluid.layers.transpose(signal_b, [0, 2, 3, 1])
    signal_a_out = fluid.layers.argmax(x=signal_a_out, axis=-1)
    signal_b_out = fluid.layers.argmax(x=signal_b_out, axis=-1)

    learning_rate = fluid.layers.piecewise_decay(boundaries=BOUNDARIES, values=VALUES)
    decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
                                               WARM_UP_STEPS,
                                               START_LR,
                                               END_LR)
    opt = fluid.optimizer.Adam(decayed_lr)
    if not FREEZE_PIX:
        loss_sum = loss_l + loss_ab
        opt.minimize(loss_l)
        opt.minimize(loss_ab)
    else:
        loss_l = loss_ab
        loss_sum = loss_ab
        opt.minimize(loss_ab)

feeder_list = [resize_l, img_l, img_l2, label_a, label_b]
train_feeder = fluid.DataFeeder(place=place,
                                feed_list=feeder_list[2:] if FREEZE_PIX else feeder_list,
                                program=train_program)
test_feeder = fluid.DataFeeder(place=place,
                               feed_list=feeder_list[2:] if FREEZE_PIX else feeder_list,
                               program=test_program)

train_loader = train_feeder.decorate_reader(fluid.io.batch(reader(TRAIN_DATA_PATH, freeze_pix=FREEZE_PIX), BATCH_SIZE),
                                            multi_devices=True)
test_loader = test_feeder.decorate_reader(fluid.io.batch(reader(TEST_DATA_PATH, is_test=True, freeze_pix=FREEZE_PIX),
                                                         BATCH_SIZE),
                                          multi_devices=True)

exe.run(start_program)

compiled_train_prog = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=loss_sum.name)
compiled_test_prog = fluid.CompiledProgram(test_program).with_data_parallel(share_vars_from=compiled_train_prog)
print("Net check --OK")
if os.path.exists(CHECK_POINT_DIR + ".pdopt") and LOAD_CHECKPOINT:
    fluid.io.load(train_program, CHECK_POINT_DIR, exe)


def if_exist(var):
    return os.path.exists(os.path.join(PER_MODEL_DIR, var.name))


if os.path.exists(PER_MODEL_DIR) and LOAD_PER_MODEL:
    fluid.io.load_vars(exe, PER_MODEL_DIR, train_program, predicate=if_exist)

MAX_ACC = 10.
ITER_NUM = 0
for epoch in range(EPOCH):
    out_loss_ab = list()
    out_loss_l = list()
    out_acc_a = list()
    out_acc_b = list()

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
                      fetch_list=[loss_ab, loss_l, decayed_lr, a_acc, b_acc])
        out_loss_ab.append(out[0][0])
        out_loss_l.append(out[1][0])
        lr = out[2]
        out_acc_a.append(out[3][0])
        out_acc_b.append(out[4][0])
        cost_time = time.time() - start_time
        if data_id % 50 == 49:
            print(epoch,
                  "-",
                  data_id,
                  "TRAIN:\t{:.6f}".format(sum(out_loss_ab) / len(out_loss_ab)),
                  "L_PSNR:{:.6f}".format(10 * np.log10(255 * 255 / sum(out_loss_l) / len(out_loss_l))),
                  "\tTIME:\t{:.4f}/s".format(cost_time / len(data)),
                  "\tLR:", lr)
            print("ACC_A:\t{:.6f}".format(np.average(out_acc_a)), "\tB:\t{:.6f}".format(np.average(out_acc_b)))
            out_loss_ab = []
            out_loss_l = []
            print("\033[0;37;41m[WARNING]\tSaving checkpoint... Please don't stop running! \033[0m")
            fluid.io.save(train_program, CHECK_POINT_DIR)
            print("\033[0;37;42m[INFO]\tDone\033[0m")
        if data_id % 500 == 500 - 1:
            out_acc_a = []
            out_acc_b = []
            out_loss_l = []
            print("Run test...")
            for t_data_id, data_t in enumerate(test_loader()):
                if t_data_id == 200:
                    break
                out = exe.run(program=compiled_test_prog,
                              feed=data_t,
                              fetch_list=[loss_l, a_acc, b_acc])
                out_loss_l.append(out[0][0])
                out_acc_a.append(out[1][0])
                out_acc_b.append(out[2][0])
            test_acc = np.average(out_acc_a) + np.average(out_acc_b)
            if test_acc >= MAX_ACC:
                MAX_ACC = test_acc
                fluid.io.save_inference_model(dirname=MODEL_DIR,
                                              feeded_var_names=["img_l2", "resize_l"],
                                              target_vars=[signal_l, signal_a_out, signal_b_out],
                                              executor=exe,
                                              main_program=train_program)

            print("\033[0;37;46m",
                  epoch,
                  "TEST: L_PSNR:{:.8f}".format(10 * np.log10(255 * 255 / sum(out_loss_l) / len(out_loss_l))),
                  "\tMAX ACC:\t{:.4f}".format(MAX_ACC))
            print("ACC_A:\t{:.6f}".format(np.average(out_acc_a)), "\tB:\t{:.6f}".format(np.average(out_acc_b)),
                  "\033[0m")
    if ITER_NUM == MAX_ITER_NUM:
        break
