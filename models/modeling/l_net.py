import paddle.fluid as fluid
from models.libs.model_libs import scope
from models.libs.model_libs import bn_relu, relu
from models.libs.model_libs import conv, max_pool


def double_conv(data, out_ch):
    with scope("conv0"):
        data = bn_relu(
            conv(data, out_ch, 3, stride=1, padding=1))
    with scope("conv1"):
        data = bn_relu(
            conv(data, out_ch, 3, stride=1, padding=1))
    return data


def down(data, out_ch):
    # 下采样：max_pool + 2个卷积
    with scope("down"):
        data = max_pool(data, 2, 2, 0)
        data = double_conv(data, out_ch)
    return data


def up(data, short_cut, out_ch, im_shape):
    with scope("up"):
        data = fluid.layers.resize_bilinear(data, out_shape=im_shape)
        data = fluid.layers.concat([data, short_cut], axis=1)
        data = double_conv(data, out_ch)
    return data


def encode(data):
    # 编码器设置
    short_cuts = []
    with scope("encode"):
        with scope("block1"):
            data = double_conv(data, 64)
            short_cuts.append(data)
        with scope("block2"):
            data = down(data, 128)
            short_cuts.append(data)
        with scope("block3"):
            data = down(data, 256)
            short_cuts.append(data)
        with scope("block4"):
            data = down(data, 512)
            short_cuts.append(data)
        with scope("block5"):
            data = down(data, 512)
    return data, short_cuts


def decode(data, short_cuts, im_shape):
    # 解码器设置，与编码器对称
    with scope("decode"):
        with scope("decode1"):
            i_shape = im_shape / 2 ** 3
            data = up(data, short_cuts[3], 256, i_shape)
        with scope("decode2"):
            i_shape = im_shape / 2 ** 2
            data = up(data, short_cuts[2], 128, i_shape)
        with scope("decode3"):
            i_shape = im_shape / 2 ** 1
            data = up(data, short_cuts[1], 64, i_shape)
        with scope("decode4"):
            i_shape = im_shape / 2 ** 0
            data = up(data, short_cuts[0], 64, i_shape)
    return data


def l_net(inp, im_shape, class_num):
    encode_data, short_cuts = encode(inp)
    decode_data = decode(encode_data, short_cuts, im_shape)
    out_data = fluid.layers.conv2d(decode_data, class_num, 1, 1)
    return out_data


if __name__ == '__main__':
    image_shape = [-1, 3, 321, 320]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    im_sha = fluid.data(name='shape', shape=[2], dtype='int64')
    logit = l_net(image, im_sha, 3)
    print("logit:", logit.shape)
