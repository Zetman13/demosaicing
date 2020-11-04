import numpy as np
from numba import njit, prange


@njit
def ppg(input_data, postprocess_iterations):
    r = np.zeros((input_data.shape[0] + 4, input_data.shape[1] + 4))
    g = np.zeros((input_data.shape[0] + 4, input_data.shape[1] + 4))
    b = np.zeros((input_data.shape[0] + 4, input_data.shape[1] + 4))

    r[2:-2, 2:-2] = input_data[:, :, 0]
    g[2:-2, 2:-2] = input_data[:, :, 1]
    b[2:-2, 2:-2] = input_data[:, :, 2]

    stage1(r, g, b)
    stage2(r, g, b)
    stage3(r, g, b)
    post_process(r, g, b, m=postprocess_iterations, k=5)

    return np.dstack((r[2:-2, 2:-2], g[2:-2, 2:-2], b[2:-2, 2:-2]))


@njit
def stage1(r, g, b):
    rb = r + b
    for i in prange(2, rb.shape[0] - 2, 2):
        for j in prange(2, rb.shape[1] - 2, 2):
            rb_pattern = get_pattern(rb, i, j, k=5)
            g_pattern = get_pattern(g, i, j, k=5)
            g[i, j] = normalize(stage1_cell(rb_pattern, g_pattern))
    for i in prange(3, rb.shape[0] - 2, 2):
        for j in prange(3, rb.shape[1] - 2, 2):
            rb_pattern = get_pattern(rb, i, j, k=5)
            g_pattern = get_pattern(g, i, j, k=5)
            g[i, j] = normalize(stage1_cell(rb_pattern, g_pattern))


@njit
def stage1_cell(rb, g):
    dn = 2 * np.abs(rb[12] - rb[2]) + np.abs(g[17] - g[8])
    de = 2 * np.abs(rb[12] - rb[14]) + np.abs(g[11] - g[13])
    dw = 2 * np.abs(rb[12] - rb[10]) + np.abs(g[13] - g[11])
    ds = 2 * np.abs(rb[12] - rb[22]) + np.abs(g[7] - g[17])

    d_min = min([dn, de, dw, ds])
    if dn == d_min:
        result = (g[7] * 3 + g[17] + rb[12] - rb[2]) / 4
    elif de == d_min:
        result = (g[13] * 3 + g[11] + rb[12] - rb[14]) / 4
    elif dw == d_min:
        result = (g[11] * 3 + g[13] + rb[12] - rb[10]) / 4
    else:
        result = (g[17] * 3 + g[7] + rb[12] - rb[22]) / 4
    return normalize(result)


@njit
def stage2(r, g, b):
    # Left and Right - Red
    for i in prange(2, r.shape[0] - 2, 2):
        for j in prange(3, r.shape[1] - 2, 2):
            r_pattern = get_pattern(r, i, j, k=3)
            g_pattern = get_pattern(g, i, j, k=3)
            b_pattern = get_pattern(b, i, j, k=3)
            r[i, j], b[i, j] = stage2_cell(r_pattern, b_pattern, g_pattern)

    # Left and Right - Blue
    for i in prange(3, r.shape[0] - 2, 2):
        for j in prange(2, r.shape[1] - 2, 2):
            r_pattern = get_pattern(r, i, j, k=3)
            g_pattern = get_pattern(g, i, j, k=3)
            b_pattern = get_pattern(b, i, j, k=3)
            b[i, j], r[i, j] = stage2_cell(b_pattern, r_pattern, g_pattern)


@njit
def stage2_cell_old(col1, col2, g):
    return normalize(hue_transit(g[11], g[12], g[13], col1[11], col1[13])), \
           normalize(hue_transit(g[7], g[12], g[17], col2[7], col2[17]))


@njit
def stage2_cell(col1, col2, g):
    return normalize(hue_transit(g[3], g[4], g[5], col1[3], col1[5])), \
           normalize(hue_transit(g[1], g[4], g[7], col2[1], col2[7]))


@njit
def stage3(r, g, b):
    for i in prange(2, r.shape[0] - 2, 2):
        for j in prange(2, r.shape[1] - 2, 2):
            r_pattern = get_pattern(r, i, j, k=5)
            g_pattern = get_pattern(g, i, j, k=5)
            b_pattern = get_pattern(b, i, j, k=5)
            b[i, j] = stage3_cell(r_pattern, b_pattern, g_pattern)
    for i in prange(3, r.shape[0] - 2, 2):
        for j in prange(3, r.shape[1] - 2, 2):
            r_pattern = get_pattern(r, i, j, k=5)
            g_pattern = get_pattern(g, i, j, k=5)
            b_pattern = get_pattern(b, i, j, k=5)
            r[i, j] = stage3_cell(b_pattern, r_pattern, g_pattern)


@njit
def stage3_cell(c1, c2, g):
    ne = np.abs(c2[8] - c2[16]) + np.abs(c1[4] - c1[12]) + np.abs(c1[12] - c1[20]) + np.abs(g[8] - g[12]) + np.abs(g[12] - g[16])
    nw = np.abs(c2[6] - c2[18]) + np.abs(c1[0] - c1[12]) + np.abs(c1[12] - c1[24]) + np.abs(g[6] - g[12]) + np.abs(g[12] - g[18])

    if ne < nw:
        return normalize(hue_transit(g[8], g[12], g[16], c2[8], c2[16]))
    else:
        return normalize(hue_transit(g[6], g[12], g[18], c2[6], c2[18]))


@njit
def post_process(r, g, b, m=1, k=5):
    for _ in range(m):
        rg = r - g
        bg = b - g
        for i in range(2, r.shape[0]-2):
            for j in range(2, r.shape[1] - 2):
                rg_med = np.median(get_pattern(rg, i, j, k=k))
                bg_med = np.median(get_pattern(bg, i, j, k=k))
                r[i, j] = normalize(rg_med + g[i, j])
                b[i, j] = normalize(bg_med + g[i, j])
                g[i, j] = normalize((- rg_med - bg_med + r[i, j] + b[i, j]) / 2)


@njit
def get_pattern(data, i, j, k):
    return data[i - (k // 2):i + (k // 2 + 1), j - (k // 2):j + (k // 2 + 1)].flatten()




@njit
def hue_transit(l1, l2, l3, v1, v3):
    if l1 < l2 < l3 or l1 > l2 > l3:
        return v1 + (v3 - v1) * (l2 - l1) / (l3 - l1)
    else:
        return (v1 + v3) / 2 + (l2 - (l1 + l3) / 2) / 2


@njit
def normalize(x):
    return max(min(round(x), 255), 0)
