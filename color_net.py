import paddle


class SampleCNNGroup(paddle.nn.Layer):
    def __init__(self, input_dim: int = 1, out_dim: int = None):
        super().__init__()
        if out_dim is None:
            out_dim = input_dim
        mid_channels = input_dim // 2
        self.group = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=input_dim,
                                                           out_channels=mid_channels,
                                                           kernel_size=3,
                                                           padding="SAME"),
                                          paddle.nn.LeakyReLU(0.1),
                                          paddle.nn.BatchNorm2D(mid_channels),
                                          paddle.nn.Conv2D(in_channels=mid_channels,
                                                           out_channels=out_dim,
                                                           kernel_size=3,
                                                           padding="SAME"),
                                          paddle.nn.LeakyReLU(0.1),
                                          paddle.nn.BatchNorm2D(out_dim))

    def forward(self, ipt):
        return self.group(ipt)


class Down(paddle.nn.Layer):
    def __init__(self, input_dim=1, out_dim=1, res_scale: float = None):
        super().__init__()
        self.res_scale = res_scale
        self.down = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=input_dim,
                                                          out_channels=out_dim,
                                                          stride=2,
                                                          kernel_size=3,
                                                          padding=1),
                                         paddle.nn.LeakyReLU(0.1),
                                         paddle.nn.BatchNorm2D(out_dim))
        self.conv = SampleCNNGroup(out_dim)

    def forward(self, ipt):
        down = self.down(ipt)
        conv = self.conv(down)
        if self.res_scale:
            conv = self.res_scale * down + conv
        return conv


class Up(paddle.nn.Layer):
    def __init__(self, input_dim=1, out_dim=1, res_scale: float = None):
        super().__init__()
        self.res_scale = res_scale
        self.up = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=input_dim,
                                                        out_channels=out_dim,
                                                        kernel_size=3,
                                                        padding="SAME"),
                                       paddle.nn.LeakyReLU(0.1),
                                       paddle.nn.Upsample(scale_factor=2))
        self.conv = SampleCNNGroup(out_dim)

    def forward(self, ipt):
        up = self.up(ipt)
        conv = self.conv(up)
        if self.res_scale:
            conv = self.res_scale * up + conv
        return conv


class Block(paddle.nn.Layer):
    def __init__(self, input_dim=1, out_dim=1, base_dim=32):
        super().__init__()
        self.decoder = paddle.nn.Sequential(Down(input_dim=input_dim, out_dim=base_dim, res_scale=0.25),
                                            Down(input_dim=1 * base_dim, out_dim=2 * base_dim, res_scale=0.5),
                                            Down(input_dim=2 * base_dim, out_dim=3 * base_dim, res_scale=0.75),
                                            Down(input_dim=3 * base_dim, out_dim=4 * base_dim, res_scale=1))
        self.encoder = paddle.nn.Sequential(Up(input_dim=4 * base_dim, out_dim=3 * base_dim, res_scale=1),
                                            Up(input_dim=3 * base_dim, out_dim=2 * base_dim, res_scale=0.75),
                                            Up(input_dim=2 * base_dim, out_dim=1 * base_dim, res_scale=0.5),
                                            Up(input_dim=base_dim, out_dim=base_dim // 2, res_scale=0.25),
                                            SampleCNNGroup(base_dim // 2))
        self.out_layer = paddle.nn.Conv2D(base_dim // 2, out_dim, 1)

    def forward(self, ipt):
        decoder = self.decoder(ipt)
        encoder = self.encoder(decoder)
        out = self.out_layer(encoder)
        return out


class ColorNet(paddle.nn.Layer):
    def __init__(self, class_dim: int):
        super().__init__()
        self.L2AB = Block(input_dim=1, out_dim=class_dim)
        self.AB2L = Block(input_dim=class_dim, out_dim=1)

    def forward(self, ipt):
        ab1 = self.L2AB(ipt)
        l2 = self.AB2L(ab1)
        with paddle.no_grad():
            ab2 = self.L2AB(l2)
        return [ipt, l2, ab2]


if __name__ == '__main__':
    model = paddle.Model(ColorNet(30))
    model.summary((8, 1, 256, 256))
