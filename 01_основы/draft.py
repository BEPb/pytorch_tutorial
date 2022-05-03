"""
Python 3.10 программа для проверки правильности предсказаний
Название файла draft.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-05-03
"""
import pandas as pd

df_true = pd.read_csv('../titanic/submission-titanic-100.csv')  # файл 100% правильными ответами для оценки наших предсказаний
df_sub = pd.read_csv('submission.csv')  # файл с предсказанными ответами

df_diff = pd.concat([df_true, df_sub]).drop_duplicates(keep=False)  # новый фрейм в котором остались только различия

print('Общее количество прогнозов', len(df_sub))
print('Количество неправильных ответов', len(df_diff)/2)
print('Процент правильных ответов', round((1 - len(df_diff)/(2 * len(df_true)))*100, 2), "%")

