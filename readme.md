# Задание №1 по обработке и анализу изображений

Реализован алгоритм демозаикинга **Patterned Pixel Gradient (PPG)**, проверено восстановление с его помощью восстановление цветное изображение (*Result.bmp*) из полутонового(*RGB_CFA.bmp*). Также была добавлен шаг постпроцессинга, описанный в статье **Adaptive Homogeneity-Directed Demosaicing Algorithm**. 

В результате работы алгоритма разрешение изображения снижается в 1.27 раза - с 14 у оригинального изображения до 11 у восстановленного.

Зависимости для запуска программы:
- Python 3.8
- numpy>=1.18.5
- numba>=0.51.2
- pillow>=8.0.1
- opencv-python>=4.2.0 

Полученные метрики алгоритма для изображения миры:

1. **Без постпроцессинга** - *Result_nopost.bmp*:
    - **Time of execution:** 16.4 seconds
    - **Speed of algorithm:** 1890 msec/megapixel
    - **MSE:** 35.46
    - **PSNR:** 32.63
    
2. **Одна итерация постпроцессинга** - *Result_post1.bmp*:
    - **Time of execution:** 31.1 seconds
    - **Speed of algorithm:** 3596 msec/megapixel
    - **MSE:** 30.26
    - **PSNR:** 33.32
    
3. **Две итерации постпроцессинга** - *Result_post2.bmp*:
    - **Time of execution:** 41.3 seconds
    - **Speed of algorithm:** 4768 msec/megapixel
    - **MSE:** 29.82
    - **PSNR:** 33.39
    
4. **Три итерации постпроцессинга** - *Result_post3.bmp*:
    - **Time of execution:** 53.6 seconds
    - **Speed of algorithm:** 6191 msec/megapixel
    - **MSE:** 29.77
    - **PSNR:** 33.39

