"""
Python 3.10 программа для изучения модели Pytorch
Название файла 05_model.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-03
"""

import torch  # библиотека pytorch
import torch.nn as nn  # библиотека нейронной сети pytorch
import torchvision  # Пакет популярных наборов данных
import torchvision.transforms as transforms

# ================================================================== #
#                         Оглавление                                 #
# ================================================================== #

# 1. Входной конвейер                                      (Line 23 to 53)
# 2. Входной конвейер для пользовательского набора данных  (Line 55 to 80)
# 3. Предварительно обученная модель                       (Line 81 to 100)
# 4. Сохранение и чтение модели                            (Line 100 to 110)

# ================================================================== #
#                         1. Входной конвейер                        #
# ================================================================== #

# Загрузите и создайте набор данных CIFAR-10.
train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# Получить одну пару данных (читать данные с диска).
image, label = train_dataset[0]
print('размер изображения = ', image.size())
print('метка', label)

# Загрузчик данных (обеспечивает очереди и потоки очень простым способом).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# Когда начинается итерация, очередь и поток начинают загружать данные из файлов.
data_iter = iter(train_loader)

# Мини-пакет изображений и меток.
images, labels = data_iter.next()

# Фактическое использование загрузчика данных показано ниже.
for images, labels in train_loader:
    # Здесь должен быть написан обучающий код.
    pass


# ================================================================== #
#     2.Входной конвейер для пользовательского набора данных         #
# ================================================================== #

# # Вы должны создать свой собственный набор данных, как показано ниже.
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # TODO
#         # 1. Инициализировать пути к файлам или список имен файлов.
#         pass
#     def __getitem__(self, index):
#         # TODO
#         # 1. Считайте данные из файла (например, с помощью numpy.fromfile, PIL.Image.open).
#         # 2. Предварительно обработайте данные (например, torchvision.Transform).
#         # 3. Вернуть пару данных (например, изображение и метку).
#         pass
#     def __len__(self):
#         # Вы должны изменить 0 на общий размер вашего набора данных.
#         return 0
#
# # Затем вы можете использовать готовый загрузчик данных.
# custom_dataset = CustomDataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
#                                            batch_size=64,
#                                            shuffle=True)

# ================================================================== #
#                        3. Предварительно обученная модель          #
# ================================================================== #

# Загрузите и загрузите предварительно обученный ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# Если вы хотите настроить только верхний слой модели, установите, как показано ниже.
for param in resnet.parameters():
    param.requires_grad = False

# Заменить верхний слой для тонкой настройки.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# прямой проход
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())     # (64, 100)

# ================================================================== #
#                      4. Сохранение и чтение модели                 #
# ================================================================== #

# Сохранение и чтение модели
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Сохранение и чтение параметров модели
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
