# Author: Acer Zhang
# Datetime: 2020/8/3 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os

import paddle.fluid as fluid

from l_net import l_net, set_name
from data_reader import get_class_num, get_resize

ROOT_PATH = "./"
CHECK_POINTS_DIR = os.path.join(ROOT_PATH, "check_model.color")
SAVE_DIR = os.path.join(ROOT_PATH, "color.model")

RESIZE = get_resize()
signal_ab_num = get_class_num()
CLASS_NUM = {"L": 1, "AB": signal_ab_num}

place = fluid.CPUPlace()
exe = fluid.Executor(place)

train_program = fluid.Program()
start_program = fluid.Program()

with fluid.program_guard(train_program, start_program):
    # 训练输入层定义
    ipt_layer_L = fluid.data(name="ipt_layer_L", shape=[-1, 1, -1, -1])
    ipt_layer_AB = fluid.data(name="ipt_layer_AB", shape=[-1, 1, -1, -1])

    # 获取shape
    im_shape_L = fluid.layers.shape(ipt_layer_L)
    im_shape_L = fluid.layers.slice(im_shape_L, axes=[0], starts=[2], ends=[4])

    im_shape_AB = fluid.layers.shape(ipt_layer_AB)
    im_shape_AB = fluid.layers.slice(im_shape_AB, axes=[0], starts=[2], ends=[4])

    set_name("L")
    signal_L = l_net(ipt_layer_L, im_shape_L, 1)
    set_name("AB")
    signal_AB = l_net(ipt_layer_AB, im_shape_AB, CLASS_NUM["AB"])

    # 获取AB结果
    signal_AB = fluid.layers.resize_nearest(signal_AB, im_shape_L)
    result_AB = fluid.layers.transpose(signal_AB, [0, 2, 3, 1])
    result_AB = fluid.layers.argmax(result_AB, axis=3)

exe.run(start_program)

var_count = 0
ignore_count = 0


def if_exist(var):
    global var_count, ignore_count
    var_result = os.path.exists(os.path.join(CHECK_POINTS_DIR, var.name))
    if var_result:
        var_count += 1
    elif "w" in var.name:
        print("Can not find:", var.name)
    else:
        ignore_count += 1
    return var_result


fluid.io.load_vars(exe, CHECK_POINTS_DIR, train_program, predicate=if_exist)
print(var_count, "组参数加载成功\t", ignore_count, "组参数被忽略，存在不可训练函数，忽略后默认在__model__中保存，不影响使用")

feed_var_names = ["ipt_layer_L", "ipt_layer_AB"]
target_vars = [signal_L, result_AB]
print("正在合并参数")
fluid.io.save_inference_model(SAVE_DIR, feed_var_names, target_vars, exe, train_program)
print("保存成功", SAVE_DIR)
