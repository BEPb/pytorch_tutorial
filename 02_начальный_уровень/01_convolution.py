"""
Python 3.10 программа для изучения сверточной нейронной сети Pytorch
Название файла 01_convolution.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-03
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Конфигурация оборудования
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Гипер параметры
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST данные
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Загрузка
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Сверточная нейронная сеть (два сверточных слоя)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Ошибка и оптимизация
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Тренировка модели
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Эпоха [{}/{}], Шаг [{}/{}], Ошибка: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Тестирование модели
model.eval()  # eval режим (batchnorm использует скользящее среднее/дисперсию вместо мини-пакетного среднего/дисперсии)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Проверка точности модели на 10000 тестовых изображений: {} %'.format(100 * correct / total))

# Сохраним контрольную точку модели
torch.save(model.state_dict(), 'model.ckpt')