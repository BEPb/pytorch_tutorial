"""
Python 3.10 программа для изучения простой нейронной сети Pytorch на примере предсказания выживших пассажиров титаника
Название файла 04_simple_neural_network.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-02
"""


'''Входной слой состоит из 8 нейронов, которые составляют входные данные в наборе данных. Затем входные данные 
передаются через два скрытых слоя, каждый из которых содержит 512 узлов, с использованием функции активации 
линейного выпрямителя (ReLU). Наконец, у нас есть выходной слой с двумя узлами, соответствующими результату.  Для 
такой задачи классификации мы будем использовать выходной слой softmax.   

### Класс для построения нейронной сети
Для создания нейросети в PyTorch используется класс nn.Module. Для его использования необходимо наследование, 
которое позволит использовать весь функционал базового класса nn.Module, но при этом еще возможно переписать базовый 
класс для построения модели или прямого прохождения по сети. Код ниже поможет объяснить это:'''

'''
Основная структура данных torch.nn — это модуль, представляющий собой абстрактное понятие, которое может представлять 
 определенный слой в нейронной сети или нейронная сеть, содержащая много слоев. На практике наиболее распространенным 
 способом является наследование nn.Module и создание собственной сети/уровня. Давайте сначала посмотрим, как 
 использовать nn.Module для реализации вашего собственного полносвязного уровня. Полносвязный слой, также известный 
 как аффинный слой.
'''

import torch  # библиотека pytorch
from torch.autograd import Variable  # импортировать переменную из библиотеки pytorch
import torch.nn as nn  # библиотека нейронной сети pytorch
import torch.nn.functional as F  # функции нейронной сети


class Net(nn.Module):  # В таком определении можно увидеть наследование базового класса nn.Module
    def __init__(self):  # инициализация класса или конструктор класса
        super(Net, self).__init__()  # функция super() создает объект базового класса

        # в следующих трех строках кода мы создаем полносвязные слои
        '''
        Полносвязный слой нейронной сети представлен объектом nn.Linear,
         в котором первым аргументом является количество узлов в i-м слое, а вторым — количество узлов в слое i+1. 
         Как видно из кода, первый слой принимает 7 узлов в качестве входных данных и подключается к первому скрытому 
         слою с 512 узлами.
        '''
        self.fc1 = nn.Linear(7, 512)

        # Далее идет подключение к другому скрытому слою с 512 узлами.
        self.fc2 = nn.Linear(512, 512)

        # И, наконец, подключение последнего скрытого слоя к выходному слою двумя узлами.
        self.fc3 = nn.Linear(512, 2)

        # Определим пропорцию или нейроны для отсева, отброса или обнуления (dropout)
        self.dropout = nn.Dropout(0.2)  # 0,2 - вероятность обнуления элемента. По умолчанию: 0,5

        '''
        После определения скелета сетевой архитектуры необходимо задать
         принципы, по которым данные будут проходить через него. Это делается с помощью метода forward().
         определяется, что переопределяет фиктивный метод в базовом классе и требует определения для каждой сети
        '''

    def forward(self, x):
        '''Для метода forward() мы принимаем входные данные x в качестве основного аргумента
         Далее загружаем все в первый полносвязный слой self.fc1(x) и применяем активацию ReLU
         функция для узлов в этом слое с помощью F.relu()'''
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

         # Из-за иерархической природы этой нейронной сети мы заменяем x на каждом этапе и отправляем
         # это на следующий слой
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Проделываем эту процедуру на трех связанных слоях, кроме последнего.
        x = self.fc3(x)

        '''На последнем слое возвращаем не ReLU, а логарифмическую функцию активации softmax.
         Это, в сочетании с отрицательной функцией потерь логарифмического правдоподобия, дает мультиклассовая
         функция потерь на основе кросс-энтропии, которую мы будем использовать для обучения сети.
        '''
        return x  # получаем бинарное предсказание

'''Мы определили нейронную сеть. Следующим шагом является создание экземпляра этой архитектуры.:'''

model = Net()
print('При выводе экземпляра класса Net получаем следующее: \n', model)
print('Что очень удобно, так как подтверждает структуру нашей нейросети.')

'''Обучение сети¶
Далее необходимо указать метод оптимизации и критерий качества:'''

import torch.optim as optim  # это пакет, реализующий различные алгоритмы оптимизации.
# Наиболее часто используемые методы уже поддерживаются, а интерфейс достаточно общий,
# чтобы в будущем можно было легко интегрировать более сложные

learning_rate = 0.01
# В первой строке мы создаем оптимизатор на основе стохастического градиентного спуска,
# установка скорости обучения (в нашем случае мы определим этот показатель равным 0,01)

# Выполняем оптимизацию стохастическим градиентным спуском
# Еще в оптимизаторе нужно определить все остальные сетевые параметры,
# но это легко делается в PyTorch благодаря методу .parameters()
# в базовом классе nn.Module, который наследуется от него в новый класс Net

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

'''
Затем устанавливается метрика контроля качества, отрицательная функция потерь логарифмического правдоподобия.
Этот тип функции в сочетании с логарифмической функцией softmax на выходе
нейронной сети, дает эквивалентную кросс-энтропийную потерю классификации.
'''

# error = nn.NLLLoss()
error = nn.CrossEntropyLoss()  # Функция потерь (перекрестная энтропия CE)

'''Внешний обучающий цикл проходит через количество эпох, а внутренний обучающий цикл проходит через все обучающие 
данные в пакетах, размер которых задается в коде как batch_size. В следующей строке мы конвертируем данные и целевую 
переменную в переменные PyTorch. Входной набор данных имеет размер (batch_size, 1, 28, 28) при извлечении из 
загрузчика данных. Такой 4D-тензор больше подходит для архитектуры сверточной нейронной сети, чем для нашей 
полносвязной сети. Однако необходимо уменьшить размерность данных с (1,28,28) до одномерного случая 
для 28 x 28 = 784 входных узлов.'''


# подключаем библиотеки
import numpy as np  # линейная алгебра
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

# Подготовим набор данных
# загрузить данные
df1 = pd.read_csv("../titanic/train.csv")

# Половое кодирование
binar = LabelBinarizer().fit(df1.loc[:, "Sex"])
df1["Sex"] = binar.transform(df1["Sex"])

# Начато кодирование
df1["Embarked"] = df1["Embarked"].fillna('S')
df_Embarked = pd.get_dummies(df1.Embarked)
df1 = pd.concat([df1, df_Embarked], axis=1)

# Семья
df1['Family'] = df1['SibSp'] + df1['Parch'] + 1
df1['Alone'] = df1['Family'].apply(lambda x: 0 if x > 1 else 1)

# Возраст
df1['Age'] = df1['Age'].fillna(-0.5)

features = ["Pclass", "Sex", "Age", "C", "Q", "S", "Alone"]

X_train = df1[features].values
y_train = df1["Survived"].values

# Масштабирование функций
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

'''Пришло время обучить нейронную сеть. Во время обучения данные будут извлекаться из объекта загрузки данных. От 
загрузчика входные и целевые данные будут приходить пачками, которые будут подаваться в нашу нейросеть и функцию 
потерь соответственно. Ниже приведен полный код для обучения:'''

#  Обучение модели
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
batch_size = 64
n_epochs = 500
batch_no = len(X_train) // batch_size

print("Количество пакетов - {}, \n количество входных данных - {}, \n количество выходных данных - {}".format(batch_no,
                                                                                                 len(X_train), len(y_train)))

train_loss = 0
train_loss_min = np.Inf

for epoch in range(n_epochs):
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size

        # Определить переменные
        x_var = Variable(torch.FloatTensor(X_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end]))

        '''В следующей строке мы запускаем optimizer.zero_grad(), который обнуляет или перезапускает градиенты в модели,
         чтобы они были готовы для дальнейшего обратного распространения. Другие библиотеки реализуют это неявно, 
         но имейте в виду, что PyTorch делает это явно.'''
        # очистка градиентов
        optimizer.zero_grad()

        # Прямое распространение
        '''В следующей строке мы подаем порцию данных на вход нашей модели, вызывает метод forward() в классе Net.'''
        output = model(x_var)

        # Рассчитать softmax и вычислить перекрестной энтропии
        # Эта история происхождения представляет собой исключительную логарифмическую исходную правдоподобию между
        # выходными данными нашей нейронной сети и истинными метками данного пакета данных.
        '''После запуска строки переменная outputs будет иметь логарифмический вывод softmax из нашей нейронной сети для
         заданный пакет данных. Это одна из замечательных особенностей PyTorch, так как вы можете активировать любой 
         стандартный отладчик Python. вы обычно используете и сразу видите, что происходит в нейронной сети. Это 
         отличается от других библиотек глубокого обучения, TensorFlow и Keras, которые требуют сложной отладки, чтобы 
         узнать, что на самом деле создает ваша нейронная сеть.
         Надеюсь, вы поиграете с кодом для этого руководства и увидите, насколько удобен отладчик PyTorch.'''
        loss = error(output, y_var)

        # Вычислить градиенты
        '''Следующая строка запускает операцию обратного распространения ошибки от переменной потерь обратно через нейронную сеть.
         Если вы сравните это с упомянутой выше операцией .backward(), которую мы рассмотрели в руководстве, вы увидите
         что в операции .backward() не используется аргумент. Скалярные переменные не требуют аргументов, когда для них используется .backward();
         только тензорам нужен соответствующий аргумент для передачи в операцию .backward().'''
        loss.backward()

        # Обновление параметров
        '''В следующей строке мы просим PyTorch выполнить пошаговый градиентный спуск на основе градиентов, вычисленных во время операции .backward().'''
        optimizer.step()

        # Получить прогнозы от максимального значения
        '''метод data.max(1), который возвращает индекс наибольшего значения в конкретном тензорном измерении.
         Теперь выходные данные нашей нейронной сети будут иметь размер (batch_size, 2), где каждое значение из второго измерения
         длины 2 — это логарифмическая вероятность, которую нейронная сеть присваивает каждому выходному классу. 
         Значение с наибольшей логарифмической вероятностью — это число от 0 до 1, которое нейронная сеть распознает 
         на входном изображении.        
        '''
        values, labels = torch.max(output, 1)
        num_right = np.sum(labels.data.numpy() == y_train[start:end])
        train_loss += loss.item() * batch_size

    train_loss = train_loss / len(X_train)

    # сохраним потери и итерации
    count += 1
    iteration_list.append(count)
    loss_list.append(train_loss)
    accuracy = num_right / len(y_train[start:end])
    accuracy_list.append(accuracy)

    if train_loss <= train_loss_min:
        print("Потеря проверки уменьшилась ({:6f} ===> {:6f}). Сохранение модели...".format(train_loss_min, train_loss))
        torch.save(model.state_dict(), "model.pt")
        train_loss_min = train_loss

    '''Наконец, мы будем печатать результаты каждый раз, когда модель достигает определенного количества эпох:
         Эта функция распечатывает наш прогресс по эпохам обучения и показывает ошибку нейросети в этот момент.'''
    if epoch % 50 == 0:
        print('Итерация: {}  Ошибка: {}  Точность: {}%  Эпоха:{}'.format(count, train_loss, accuracy, epoch))

print('Обучение завершено!')

print("Количество правильных ответов в пакете:", num_right)
print("Длинна пакета:", len(y_train[start:end]))

# визуализация ошибки
plt.plot(iteration_list, loss_list)
plt.xlabel("Количество итераций")
plt.ylabel("Ошибка")
plt.title("Базовый класс: потери против количества итераций")
plt.show()

# визуализация точности
plt.plot(iteration_list, accuracy_list, color = "red")
plt.xlabel("Количество итераций")
plt.ylabel("Точность")
plt.title("Базовый класс: точность против количества итераций")
plt.show()

'''Отлично, мы научились создавать и обучать нашу базовую модель!
Однако мы шли к этому медленно и размеренно, разбираясь в каждом шаге. Вот как это должно быть сделано. Не нужно 
бездумно копировать код, нужно понимать, что это за код и что он делает. 
'''

# Предсказание на данных которые не видела наша модель
df_sub = pd.read_csv('../titanic/gender_submission.csv')  # файл гендерной принадлежности каждого пассажира

# Подготовим набор данных
# загрузить данные
df_test = pd.read_csv('../titanic/test.csv')  # считаем из файла данные для предсказания

# Половое кодирование
binar = LabelBinarizer().fit(df_test.loc[:, "Sex"])
df_test["Sex"] = binar.transform(df_test["Sex"])

# Начато кодирование
df_test["Embarked"] = df_test["Embarked"].fillna('S')
df_Embarked = pd.get_dummies(df_test.Embarked)
df_test = pd.concat([df_test, df_Embarked], axis=1)

# Семья
df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1
df_test['Alone'] = df_test['Family'].apply(lambda x: 0 if x > 1 else 1)

# Возраст
df_test['Age'] = df_test['Age'].fillna(-0.5)

features = ["Pclass", "Sex", "Age", "C", "Q", "S", "Alone"]

X_test = df_test[features].values

# Масштабирование функций
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=False)
with torch.no_grad():
    test_result = model(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()
print("Датафрейм предсказания выживших: \n", survived)

# Сохраним результат предсказания в файл
submission = pd.DataFrame({'PassengerId': df_sub['PassengerId'], 'Survived': survived})
print("Датафрейм номер пассажира и его предсказания выживания: \n", submission)
submission.to_csv('submission.csv', index=False)

# а теперь давайте проверим насколько точно наша модель осуществляет предсказания

df_true = pd.read_csv('../titanic/submission-titanic-100.csv')  # файл 100% правильными ответами для оценки наших предсказаний
df_sub = pd.read_csv('submission.csv')  # файл с предсказанными ответами

df_diff = pd.concat([df_true, df_sub]).drop_duplicates(keep=False)  # новый фрейм в котором остались только различия

print('Общее количество прогнозов', len(df_sub))
print('Количество неправильных ответов', len(df_diff)/2)
print('Процент правильных ответов', round((1 - len(df_diff)/(2 * len(df_true)))*100, 2), "%")
