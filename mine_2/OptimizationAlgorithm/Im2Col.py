import numpy as np
def im2col(img, k_h, k_w, s_h, s_w, p_h, p_w):
    n, c, h, w = img.shape
    img = np.pad(img, [(0,0), (0, 0), (p_h, p_h), (p_w, p_w)], 'constant')

    out_h = (h + 2*p_h - k_h) // s_h + 1
    out_w = (w + 2*p_w - k_w) // s_w + 1
    col = np.ndarray((n, c, k_h, k_w, out_h, out_w), dtype=img.dtype)
    for y in range(k_h):
        y_lim = y + s_w * out_w
        for x in range(k_w):
            x_lim = x + s_w * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_lim:s_h, x:x_lim:s_w]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n*out_h*out_w, -1)

    return col