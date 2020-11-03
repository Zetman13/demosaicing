# Задание №1 по обработке и анализу изображений

Реализован алгоритм демозаикинга **Patterned Pixel Gradient (PPG)**, проверено восстановление с его помощью восстановление цветное изображение (*Result.bmp*) из полутонового(*RGB_CFA.bmp*).

В результате работы алгоритма разрешение изображения снижается в 1.27 раза - с 14 у оригинального изображения до 11 у восстановленного.

Зависимости для запуска программы:
- Python 3.8
- numpy>=1.18.5
- numba>=0.51.2
- pillow>=8.0.1
- opencv-python>=4.2.0 

Полученные метрики алгоритма для изображения миры:
- MSE = 36.54
- PSNR = 32.50
- Time of Execution = 11.8 секунд
- Speed = 1366.44 msec/megapixel
