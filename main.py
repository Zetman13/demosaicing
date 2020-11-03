import numpy as np
from PIL import Image
from src.ppg import ppg
from src.metrics import mse, psnr

import time


input_path = 'images/RGB_CFA.bmp'
target_path = 'images/Original.bmp'

input_data = np.asarray(Image.open(input_path), dtype=np.float32)
target_data = np.asarray(Image.open(target_path), dtype=np.float32)

start = time.time()
result_data = ppg(input_data)
exec_time = time.time() - start

print(f'Time of execution: {exec_time} seconds')
print(f'Speed of algorithm: {exec_time / (result_data.shape[0] * result_data.shape[1]) * 1e6 * 1e3} msec/megapixel')
print(f'MSE: {mse(result_data, target_data)}')
print(f'PSNR: {psnr(result_data, target_data)}')


im = Image.fromarray(result_data.astype(np.uint8))
im.save('images/Result.bmp')
