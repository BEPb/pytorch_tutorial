"""
Python 3.10 программа для изучения логистической регрессии Pytorch
Название файла 06_linear.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-03
"""

import torch  # библиотека pytorch
import torch.nn as nn  # библиотека нейронной сети pytorch
import torchvision  # Пакет популярных наборов данных
import torchvision.transforms as transforms


# Гипер-параметры
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# набор данных MNIST (рисунки и метки)
train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Загрузчик данных (входной конвейер)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Модель логистической регрессии
model = nn.Linear(input_size, num_classes)


# Ошибка и оптимизатор
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Тренировка модели
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Изменение размера изображения (batch_size, input_size)
        images = images.reshape(-1, input_size)
        
        # Проход вперед
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Эпоха [{}/{}], Шаг [{}/{}], Ошибка: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Протестировать модель
# На этапе тестирования нам не нужно вычислять градиенты (для эффективности использования памяти)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Точность модели на 10000 тестовых изображений: {} %'.format(100 * correct / total))

# Сохраните контрольную точку модели
torch.save(model.state_dict(), 'model.ckpt')
