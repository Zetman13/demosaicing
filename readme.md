# Задание №1 по обработке и анализу изображений

Реализован алгоритм демозаикинга **Patterned Pixel Gradient (PPG)**, проверено восстановление с его помощью восстановление цветное изображение (*Result.bmp*) из полутонового(*RGB_CFA.bmp*).
Для работы с граничными пикселями применялся паддинг пикселями с нулевыми значениями, благодаря чему не происходит понижение разрешения изображения.

Зависимости для запуска программы:
- Python 3.8
- numpy>=1.18.5
- numba>=0.51.2
- pillow>=8.0.1

Полученные метрики алгоритма для изображения миры:
- MSE = 489.27
- PSNR = 21.24
- Time of Execution = 7.7 секунд
- Speed = 889.25 msec/megapixel
