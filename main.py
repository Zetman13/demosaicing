import numpy as np
from PIL import Image
from numba import njit

import time


R, G, B = 0, 1, 2


@njit
def ppg(bayer_data):
    global R, G, B
    data = np.zeros((3, bayer_data.shape[0] + 4, bayer_data.shape[1] + 4))

    # Step 0. Copy known color values to 3D array
    init_colors(data, bayer_data)

    # Stage 1
    stage1(data, R)  # Red Cycle
    stage1(data, B)  # Blue Cycle

    # Stage 2
    stage2(data, R, B)
    stage2(data, B, R)

    # Stage 3.
    # Red Cycle
    stage3(data, R, B)
    stage3(data, B, R)

    return data[:, 2:-2, 2:-2].transpose(1, 2, 0).astype(np.uint8)


@njit
def init_colors(data, bayer_data):
    global R, G, B
    for i in range(bayer_data.shape[0]):
        for j in range(bayer_data.shape[1]):
            if i % 2 == 0 and j % 2 == 0:
                data[R, i + 2, j + 2] = bayer_data[i, j]
            elif i % 2 != 0 and j % 2 != 0:
                data[B, i + 2, j + 2] = bayer_data[i, j]
            else:
                data[G, i + 2, j + 2] = bayer_data[i, j]


@njit
def stage1(data, color):
    global R, G, B
    loop_start = 2 if color == R else 3
    for i in range(loop_start, data.shape[1] - 2, 2):
        for j in range(loop_start, data.shape[2] - 2, 2):
            data[G, i, j] = normalize(stage1_cell(data, i, j, color))


@njit
def stage1_cell(data, i, j, color):
    global R, G, B
    dn = 2 * np.abs(data[color, i, j] - data[color, i - 2, j]) + np.abs(data[G, i + 1, j] - data[G, i - 1, j])
    de = 2 * np.abs(data[color, i, j] - data[color, i, j + 2]) + np.abs(data[G, i, j + 1] - data[G, i, j - 1])
    dw = 2 * np.abs(data[color, i, j] - data[color, i, j - 2]) + np.abs(data[G, i, j + 1] - data[G, i, j - 1])
    ds = 2 * np.abs(data[color, i, j] - data[color, i + 2, j]) + np.abs(data[G, i + 1, j] - data[G, i - 1, j])

    d_min = np.argmin(np.array([dn, de, dw, ds]))
    if d_min == 0:
        return (data[G, i - 1, j] * 3 + data[G, i + 1, j] + data[color, i, j] - data[color, i - 2, j]) / 4
    elif d_min == 1:
        return (data[G, i, j + 1] * 3 + data[G, i, j - 1] + data[color, i, j] - data[color, i, j + 2]) / 4
    elif d_min == 2:
        return (data[G, i, j - 1] * 3 + data[G, i, j + 1] + data[color, i, j] - data[color, i, j - 2]) / 4
    else:
        return (data[G, i + 1, j] * 3 + data[G, i - 1, j] + data[color, i, j] - data[color, i + 2, j]) / 4


@njit
def stage2(data, color1, color2):
    global R, G, B
    start1, start2 = (2, 3) if color1 == R else (3, 2)
    for i in range(start1, data.shape[1] - 2, 2):
        for j in range(start2, data.shape[2] - 2, 2):
            data[color1, i - 2, j - 2] = normalize(hue_transit(
                l1=data[G, i, j - 1],
                l2=data[G, i, j],
                l3=data[G, i, j + 1],
                v1=data[color1, i, j - 1],
                v3=data[color1, i, j + 1]
            ))
            data[color2, i - 2, j - 2] = normalize(hue_transit(
                l1=data[G, i - 1, j],
                l2=data[G, i, j],
                l3=data[G, i + 1, j],
                v1=data[color2, i - 1, j],
                v3=data[color2, i + 1, j]
            ))


@njit
def stage3(data, color1, color2):
    loop_start = 2 if color1 == R else 3
    for i in range(loop_start, data.shape[1] - 2, 2):
        for j in range(loop_start, data.shape[2] - 2, 2):
            data[color2, i, j] = normalize(stage3_cell(data, i, j, color1, color2))

    return data[:, 2:-2, 2:-2].transpose(1, 2, 0).astype(np.uint8)


@njit
def stage3_cell(data, i, j, color1, color2):
    global R, G, B
    ne_own = np.abs(data[color1, i, j] - data[color1, i - 1, j + 1]) + np.abs(data[color1, i, j] - data[color1, i + 1, j - 1])
    ne_other = np.abs(data[color2, i + 1, j - 1] - data[color2, i - 1, j + 1])
    ne_green = np.abs(data[G, i, j] - data[G, i - 1, j + 1]) + np.abs(data[G, i, j] - data[G, i - 1, j + 1])
    ne = ne_own + ne_other + ne_green

    nw_own = np.abs(data[color1, i, j] - data[color1, i - 1, j - 1]) + np.abs(data[color1, i, j] - data[color1, i + 1, j + 1])
    nw_other = np.abs(data[color2, i - 1, j - 1] - data[color2, i + 1, j + 1])
    nw_green = np.abs(data[G, i, j] - data[G, i - 1, j - 1]) + np.abs(data[G, i, j] - data[G, i + 1, j + 1])
    nw = nw_own + nw_other + nw_green

    if ne < nw:
        return hue_transit(
            l1=data[G, i - 1, j + 1],
            l2=data[G, i, j],
            l3=data[G, i + 1, j - 1],
            v1=data[color2, i - 1, j + 1],
            v3=data[color2, i + 1, j - 1]
        )
    else:
        return hue_transit(
            l1=data[G, i - 1, j - 1],
            l2=data[G, i, j],
            l3=data[G, i + 1, j + 1],
            v1=data[color2, i - 1, j - 1],
            v3=data[color2, i + 1, j + 1]
        )


@njit
def hue_transit(l1, l2, l3, v1, v3):
    if l1 < l2 < l3 or l1 > l2 > l3:
        return v1 + (v3 - v1) * (l2 - l1) / (l3 - l1)
    else:
        return (v1 + v3) / 2 + (l2 - (l1 + l3) / 2) / 2


@njit
def normalize(x):
    return max(min(x, 255), 0)


def mse(x1, x2):
    k_r = 0.299
    k_b = 0.114
    y1 = k_r * x1[:, :, 0] + (1 - k_r - k_b) * x1[:, :, 1] + k_b * x1[:, :, 2]
    y2 = k_r * x2[:, :, 0] + (1 - k_r - k_b) * x2[:, :, 1] + k_b * x2[:, :, 2]
    return np.square(y1-y2).mean()


input_path = 'RGB_CFA.bmp'
target_path = 'Original.bmp'

input_data = np.asarray(Image.open(input_path), dtype=np.float32)
if len(input_data.shape) == 3 and input_data.shape[2] == 3:
    input_data = np.sum(input_data, axis=2)
target_data = np.asarray(Image.open(target_path), dtype=np.float32)

start = time.time()
result = ppg(input_data)
exec_time = time.time() - start
mse_result = mse(result, target_data)

print(f'Time of execution: {exec_time} seconds')
print(f'Speed of algorithm: {exec_time / (result.shape[0] * result.shape[1]) * 1e6 * 1e3} msec/megapixel')
print(f'MSE: {mse_result}')
print(f'PSNR: {10*np.log10(255**2/mse_result)}')

im = Image.fromarray(result)
im.save('Result.bmp')
