"""
Python 3.10 программа для изучения тензоров
Название файла 01_tensor.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-02
"""
import numpy as np  # линейная алгебра
import torch  # библиотека pytorch

'''В pytorch матрица (массив) называется тензором. 3*3 матрица - это тензор 3x3.'''
print("\n 01. Изучаем тензоры \n")
array = [[1, 2, 3], [4, 5, 6]]  # создадим список списков (два списка по три числа в каждом)
first_array = np.array(array)  # Мы создаем массив numpy с помощью метода np.numpy() -> массив 2x3
print("Тип таблицы: {}".format(type(first_array)))  # Type(): тип массива. В этом примере это numpy
print("Размер таблицы: {}".format(np.shape(first_array)))  # np.shape(): форма массива. Строка х столбец
print('\n  Содержимое нашего массива:  \n', first_array)  # выведем содержимое массива на экран




tensor = torch.Tensor(2, 3)  # массив pytorch

'''Этот код создает тензор размера (2,3), заполненный нулями. В этом примере
первое число — количество строк, второе — количество столбцов.'''
print("\n 02. Изучаем тензоры torch \n")
print("Тип массива: {}".format(tensor.type))  # тип
print("Размер массива: {}".format(tensor.shape))  # размер
print('\n  Содержимое нашего массива:  \n', tensor)

'''Мы также можем создать тензор, заполненный случайными числами с плавающей запятой:'''
x = torch.rand(2, 3)
print('\n  Содержимое случайного тензора:  \n', x)


'''Умножение тензоров, добавление друг к другу и другие алгебраические операции просты:'''
x = torch.ones(2, 3)  # тензор заполнен 1
print('\n\n Тензор единиц: \n', x)
y = torch.ones(2, 3) * 2  # тензор заполнен 2
print('\n\n Результат умножения тензора единиц на 2: \n', y)
print('\n\n Результат сложения тензоров: \n', x + y)  # результат добавления тензоров


'''Распределение является одним из наиболее часто используемых методов кодирования. Поэтому давайте узнаем, 
как сделать это с помощью pytorch. Чтобы узнать, сравните numpy и tensor 
np.ones() = torch.ones()
np.random.rand() = torch.rand()
'''

# numpy единицы
print("\nNumpy {}\n".format(np.ones((2, 3))))

# pytorch единицы
print("\ntorch \n", torch.ones((2, 3)))

# numpy случайные числа
print("\nNumpy {}\n".format(np.random.rand(2, 3)))

# pytorch случайные числа
print("\ntorch \n", torch.rand(2, 3))

'''Даже если я использую pytorch для нейронных сетей, я чувствую себя лучше, если использую numpy. Поэтому обычно 
результат нейронной сети, которая является тензорной, преобразуется в массив numpy для визуализации или изучения. 
Давайте посмотрим на преобразование между тензорными и пустыми массивами.  

torch.from_numpy(): из numpy в tensor
numpy(): из tensor в numpy'''
# случайный массив numpy
array = np.random.rand(2, 2)
print("{} {}\n".format(type(array), array))

# из массива numpy в тензор
from_numpy_to_tensor = torch.from_numpy(array)
print("{}\n".format(from_numpy_to_tensor))

# из tensor в numpy
tensor = from_numpy_to_tensor
from_tensor_to_numpy = tensor.numpy()
print("{} {}\n".format(type(from_tensor_to_numpy), from_tensor_to_numpy))


print('До преобразования: \n', y)
'''Также доступна работа с функцией частичного преобразования тензора. Например y[:,1]:'''
y[:, 1] = y[:, 1] + 1
print('\n После преобразования: \n', y)
