"""
Python 3.10 программа для изучения базовой математики с Pytorch
Название файла 02_BasicMath.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-02
"""

import numpy as np  # линейная алгебра
import torch  # библиотека pytorch

'''
Базовая математика с Pytorch
Изменить размер: view()
a и b являются тензорными.
Дополнение: torch.add(a,b) = a + b
Вычитание: a.sub(b) = a - b
Поэлементное умножение: torch.mul(a,b) = a * b
Поэлементное деление: torch.div(a,b) = a / b
Среднее значение: a.mean()
Стандартное отклонение (стандартное): a.std()
'''
# создадим тензор 3х3 заполненный единицами
tensor = torch.ones(3, 3)
print("тензор 3х3 заполненный единицами\n", tensor)

# изменим размер существующего тензора на 9х1
print("Изменим размер из 3х3 на 9х1 {}{}\n".format(tensor.view(9).shape, tensor.view(9)))

# Сложение
print("Сложение двух единичных тензоров: {}\n".format(torch.add(tensor, tensor)))

# Вычитание
print("Вычитания: {}\n".format(tensor.sub(tensor)))

# Умножение
print("Умножение: {}\n".format(torch.mul(tensor, tensor)))

# Деление
print("Деление: {}\n".format(torch.div(tensor, tensor)))

# Среднее значение
tensor = torch.Tensor([1, 2, 3, 4, 5])
print("Mean: {}".format(tensor.mean()))

# Стандартное отклонение
print("Стандартное отклонение: {}".format(tensor.std()))