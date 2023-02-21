import numpy as np
from Im2Col import im2col
class pooling:
    def __init(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    def forward(self, x):
        n, c, h, w = x.shape
        out_h = int((h - self.pool_h) / self.stride + 1)
        out_w = int((w - self.pool_w) / self.stride + 1)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.stride, self.pad, self.pad)

        out = np.max(col, axis=1)

        out = out.reshape(n, out_h, out_w, x).transpose(0, 3, 1, 2)

        return out