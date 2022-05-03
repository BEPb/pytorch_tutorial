"""
Python 3.10 программа для изучения линейной регрессии Pytorch
Название файла 06_linear.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-03
"""

import torch  # библиотека pytorch
import torch.nn as nn  # библиотека нейронной сети pytorch
import numpy as np  # линейная алгебра
import matplotlib.pyplot as plt


# Гипер-параметры
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Создаем искусственный датасет
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Модель линейной регрессии
model = nn.Linear(input_size, output_size)

# ошибка и оптимизация
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Тренировка модели
for epoch in range(num_epochs):
    # Преобразование массивов numpy в тензоры факела
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Проход вперед
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print('Эпоха [{}/{}], Ошибка: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Строим график
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Оригинальные данные')
plt.plot(x_train, predicted, label='Линия предсказания')
plt.legend()
plt.show()

# Сохраняем модель
torch.save(model.state_dict(), 'model.ckpt')