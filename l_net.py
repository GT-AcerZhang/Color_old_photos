import paddle.fluid as fluid

GLOBAL_NAME = "None"
GLOBAL_ID = 0


def set_name(name):
    global GLOBAL_NAME, GLOBAL_ID
    GLOBAL_NAME = name + "_"
    GLOBAL_ID = 0


def get_conv_param():
    param_attr = fluid.ParamAttr(
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.33))
    return param_attr


def get_i_n_leaky_relu(ipt):
    global GLOBAL_ID
    GLOBAL_ID += 1
    scale_param = fluid.ParamAttr(
        name=GLOBAL_NAME + "I_N_W_" + str(GLOBAL_ID),
        initializer=fluid.initializer.Constant(1.0),
        trainable=True)
    offset_param = fluid.ParamAttr(
        name=GLOBAL_NAME + "I_N_B_" + str(GLOBAL_ID),
        initializer=fluid.initializer.Constant(0.0),
        trainable=True)
    i_n = fluid.layers.instance_norm(ipt,
                                     param_attr=scale_param,
                                     bias_attr=offset_param,
                                     name=GLOBAL_NAME + "I_N_" + str(GLOBAL_ID))
    return fluid.layers.leaky_relu(i_n, name=GLOBAL_NAME + "leaky_relu" + str(GLOBAL_ID))


def get_bn_leaky_relu(ipt):
    global GLOBAL_ID
    GLOBAL_ID += 1
    param_attr = fluid.ParamAttr(
        name=GLOBAL_NAME + "B_N_W_" + str(GLOBAL_ID),
        initializer=fluid.initializer.Normal(
            loc=1.0, scale=0.02))
    bias_attr = fluid.ParamAttr(
        name=GLOBAL_NAME + "B_N_B_" + str(GLOBAL_ID),
        initializer=fluid.initializer.Constant(value=0.0))
    b_n = fluid.layers.batch_norm(ipt,
                                  param_attr=param_attr,
                                  bias_attr=bias_attr,
                                  name=GLOBAL_NAME + "B_N_" + str(GLOBAL_ID))
    return fluid.layers.leaky_relu(b_n, name=GLOBAL_NAME + "leaky_relu" + str(GLOBAL_ID))


def conv_x3(data, out_ch):
    global GLOBAL_ID
    GLOBAL_ID += 1
    data = get_bn_leaky_relu(fluid.layers.conv2d(data,
                                                 out_ch,
                                                 3,
                                                 stride=1,
                                                 padding=1,
                                                 param_attr=get_conv_param(),
                                                 name=GLOBAL_NAME + "CONV1_" + str(GLOBAL_ID)))
    data_1 = get_bn_leaky_relu(fluid.layers.conv2d(data,
                                                   out_ch,
                                                   3,
                                                   stride=1,
                                                   padding=1,
                                                   param_attr=get_conv_param(),
                                                   name=GLOBAL_NAME + "CONV2_" + str(GLOBAL_ID)))
    # data_2 = get_bn_leaky_relu(fluid.layers.conv2d(data_1,
    #                                                out_ch,
    #                                                3,
    #                                                stride=1,
    #                                                padding="SAME",
    #                                                dilation=3,
    #                                                param_attr=get_conv_param(),
    #                                                name=GLOBAL_NAME + "CONV3_" + str(GLOBAL_ID)))
    # data = fluid.layers.concat([data_1, data_2], axis=1)
    # return data
    return data_1


def down(data, out_ch):
    global GLOBAL_ID
    GLOBAL_ID += 1
    data = fluid.layers.pool2d(data, 2, "max", 2, 0, name=GLOBAL_NAME + "POOL_" + str(GLOBAL_ID))
    data = conv_x3(data, out_ch)
    return data


def up(data, short_cut, out_ch, im_shape):
    global GLOBAL_ID
    GLOBAL_ID += 1
    data = fluid.layers.resize_nearest(data, out_shape=im_shape, name=GLOBAL_NAME + "POOL_" + str(GLOBAL_ID))
    data = fluid.layers.concat([data, short_cut], axis=1, name=GLOBAL_NAME + "CONCAT_" + str(GLOBAL_ID))
    data = conv_x3(data, out_ch)
    return data


def encode(data):
    # 编码器设置
    short_cuts = []
    data = conv_x3(data, 128)
    short_cuts.append(data)
    data = down(data, 128)
    short_cuts.append(data)
    data = down(data, 128)
    short_cuts.append(data)
    data = down(data, 256)
    short_cuts.append(data)
    data = down(data, 256)
    return data, short_cuts


def decode(data, short_cuts, im_shape):
    # 解码器设置，与编码器对称
    i_shape = im_shape / 2 ** 3
    data = up(data, short_cuts[3], 256, i_shape)
    i_shape = im_shape / 2 ** 2
    data = up(data, short_cuts[2], 128, i_shape)
    i_shape = im_shape / 2 ** 1
    data = up(data, short_cuts[1], 128, i_shape)
    i_shape = im_shape / 2 ** 0
    data = up(data, short_cuts[0], 128, i_shape)
    return data


def l_net(inp, im_shape, class_num):
    encode_data, short_cuts = encode(inp)
    decode_data = decode(encode_data, short_cuts, im_shape)
    out_data = fluid.layers.conv2d(decode_data, class_num, 1, 1, name=GLOBAL_NAME + "OUT_LAYER")
    return out_data


if __name__ == '__main__':
    image_shape = [-1, 3, 321, 320]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    im_sha = fluid.data(name='shape', shape=[2], dtype='int64')
    logit = l_net(image, im_sha, 3)
    print("logit:", logit.shape)
