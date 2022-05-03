"""
Python 3.10 программа для изучения нейронная сеть с прямой связью Pytorch
Название файла 08_feedforward.py

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
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST данные
train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Загрузка данных
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Полносвязная нейронная сеть с одним скрытым слоем
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Ошибка и оптимизация
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Тренировка модели
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Переместите тензоры на настроенное устройство
        images = images.reshape(-1, 28*28).to(device)
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

# Протестировать модель
# На этапе тестирования нам не нужно вычислять градиенты (для эффективности использования памяти)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Точность сети на 10000 тестовых изображений: {} %'.format(100 * correct / total))

# Сохраним контрольную точку модели
torch.save(model.state_dict(), 'model.ckpt')