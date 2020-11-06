import paddle


class Down(paddle.nn.Layer):
    def __init__(self, input_dim=1, out_dim=1):
        super().__init__()
        self.down = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=input_dim,
                                                          out_channels=input_dim,
                                                          kernel_size=3,
                                                          padding="SAME"),
                                         paddle.nn.LeakyReLU(0.1),
                                         paddle.nn.Conv2D(in_channels=input_dim,
                                                          out_channels=out_dim,
                                                          stride=2,
                                                          kernel_size=3,
                                                          padding=1),
                                         paddle.nn.LeakyReLU(0.1))

    def forward(self, ipt):
        down = self.down(ipt)
        return down


class Up(paddle.nn.Layer):
    def __init__(self, input_dim=1, out_dim=1):
        super().__init__()
        self.up = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=input_dim,
                                                        out_channels=input_dim,
                                                        kernel_size=3,
                                                        padding="SAME"),
                                       paddle.nn.LeakyReLU(0.1),
                                       paddle.nn.Upsample(scale_factor=2),
                                       paddle.nn.Conv2D(in_channels=input_dim,
                                                        out_channels=out_dim,
                                                        kernel_size=1),
                                       paddle.nn.LeakyReLU(0.1))

    def forward(self, ipt):
        up = self.up(ipt)
        return up


class Block(paddle.nn.Layer):
    def __init__(self, input_dim=1, out_dim=1, base_dim=32):
        super().__init__()
        self.decoder = paddle.nn.Sequential(Down(input_dim, base_dim),
                                            Down(1 * base_dim, 2 * base_dim),
                                            Down(2 * base_dim, 3 * base_dim),
                                            Down(3 * base_dim, 4 * base_dim))
        self.encoder = paddle.nn.Sequential(Up(4 * base_dim, 3 * base_dim),
                                            Up(3 * base_dim, 2 * base_dim),
                                            Up(2 * base_dim, 1 * base_dim),
                                            Up(base_dim, out_dim))

    def forward(self, ipt):
        tmp = self.decoder(ipt)
        tmp = self.encoder(tmp)
        return tmp


class ColorNet(paddle.nn.Layer):
    def __init__(self, class_dim: int):
        super().__init__()
        self.L2AB = Block(input_dim=1, out_dim=class_dim)
        self.AB2L = Block(input_dim=class_dim, out_dim=1)

    def forward(self, ipt):
        ab1 = self.L2AB(ipt)
        l2 = self.AB2L(ab1)
        ab2 = self.L2AB(l2)
        return [ipt, l2, ab2]


if __name__ == '__main__':
    model = paddle.Model(ColorNet(30))
    model.summary((1, 1, 256, 256))
