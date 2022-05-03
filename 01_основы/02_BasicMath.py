"""
Python 3.10 программа для изучения базовой математики с Pytorch
Название файла 02_BasicMath.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-02
"""

import torch  # библиотека pytorch

# ================================================================== #
#                         Оглавление                                 #
# ================================================================== #

# 1. Пример 1 (базовые операции)                  (Line 21 to 60)
# 2. Пример 2 (математические операции)           (Line 62 to 74)
# 3. Пример 3 (частичное преобразование)          (Line 75 to 70)


# ================================================================== #
#                     1. Пример 1 (базовые операции)                 #
# ================================================================== #

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
print("Сложение двух единичных тензоров: \n", torch.add(tensor, tensor))

# Вычитание
print("Вычитание: \n", tensor.sub(tensor))

# Умножение
print("Умножение: \n", torch.mul(tensor, tensor))

# Деление
print("Деление: \n", torch.div(tensor, tensor))

# Среднее значение
tensor = torch.Tensor([1, 2, 3, 4, 5])
print("Среднее значение: ", tensor.mean())

# Стандартное отклонение
print("Стандартное отклонение: ", tensor.std())

# ================================================================== #
#                     2. Пример 2 (математические операции)          #
# ================================================================== #

'''Умножение тензоров, добавление друг к другу и другие алгебраические операции:'''
x = torch.ones(2, 3)  # тензор заполнен 1
print('\n\n Тензор единиц: \n', x)
y = torch.ones(2, 3) + 2  # тензор заполнен 3
print('\n\n Результат умножения тензора единиц на 10: \n', x * 10)
print('\n\n Результат сложения тензора единиц к 2: \n', y)
print('\n\n Результат сложения двух тензоров: \n', x + y)  # результат добавления тензоров
print('\n\n Результат вычитания двух тензоров: \n', x - y)  # результат вычитания тензоров

# ================================================================== #
#                     3. Пример 3 (частичное преобразование)         #
# ================================================================== #

y = torch.ones(2, 3)   # тензор заполнен 1
print('\n До преобразования: \n', y)
'''Также доступна работа с функцией частичного преобразования тензора. Например y[:,1] преобразует вторую колонку:'''
y[:, 1] = y[:, 1] + 1
print('\n После преобразования: \n', y)

y = torch.zeros(5, 5)   # тензор заполнен 0
print('\n До преобразования: \n', y)
'''Также доступна работа с функцией частичного преобразования тензора. Например y[:,-2] преобразует предпоследнюю 
колонку:'''
y[:, -2] = y[:, -2] + 7
print('\n После преобразования: \n', y)

y = torch.zeros(8, 8)   # тензор заполнен 0
print('\n До преобразования: \n', y)
'''Также доступна работа с функцией частичного преобразования тензора. Например y[:3,:] преобразует первые три 
строки:'''

y[:3,:] = y[:3,:] + 7
print('\n После преобразования: \n', y)