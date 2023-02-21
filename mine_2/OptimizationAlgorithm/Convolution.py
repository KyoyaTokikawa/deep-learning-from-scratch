class convolution:
    def __init(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    def forward(self, x):
        k_n, c, k_h, k_w = self.W.shape
        n, c, h, w = x.shape
        out_h = int((h + 2*self.pad - k_h) / self.stride + 1)
        out_w = int((w + 2*self.pad - k_w) / self.stride + 1)
        col = im2col(x, k_h, k_w, self.stride, self.stride, ,self.pad, self.pad)
        col_W = self.W.reshape(k_n, -1).T
        out = no.dot(col, col_W) + self.b
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
