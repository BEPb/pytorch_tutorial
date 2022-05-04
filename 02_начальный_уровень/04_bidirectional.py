"""
Python 3.10 программа для изучения двунаправленной рекурентной нейронной сети Pytorch
Название файла 04_bidirectional.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-03
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Конфигурация оборудования
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Гипер-параметры
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST данные
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# загрузка
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Двунаправленная рекуррентная нейронная сеть (многие к одному)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 для двунаправленного
    
    def forward(self, x):
        # Установим начальные состояния
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 для двунаправленного
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Вперед распространять LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: размер тензора (batch_size, seq_length, hidden_size*2)
        
        # Декодировать скрытое состояние последнего временного шага
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Ошибка и оптимизация
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Тренировка модели
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Эпоха [{}/{}], Шаг [{}/{}], Ошибка: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Тест модели
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Проверка точности модели на 10000 тестовых изображений: {} %'.format(100 * correct / total))

# Сохраним контрольную точку модели
torch.save(model.state_dict(), 'model.ckpt')