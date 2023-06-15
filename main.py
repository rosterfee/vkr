import numpy as np
from pandas import DataFrame

from model import get_model_score, predict_class
from dataframe import save_the_most_important_features, whitney_feature_selection, split_dataframe, gen_dataset, \
    save_dataframe, get_sample_from_files

# column_names = ['cfs', 'SI-a', 'LF-a', 'HF-a', 'LF/HF-a', 'VLF-a', 'TP-a', 'SI-b', 'LF-b', 'HF-b', 'LF/HF-b',
#                     'VLF-b', 'TP-b', 'SI-c', 'LF-c', 'HF-c', 'LF/HF-c', 'VLF-c', 'TP-c']

# # Генерация датасета
# dataset = gen_dataset()
# # Генерация датафрейма из датасета
# df = DataFrame.from_dict(dataset, orient='index', columns=column_names)

# healthy = df[df['cfs'] == 0]
# cfs = df[df['cfs'] == 1]
# for column in df:
#     print(column)
#     print(f'healthy: {np.average(healthy[column])}')
#     print(f'cfs: {np.average(cfs[column])}')
#     print()

# # Удаление признаков, однородных по выборкам
# not_important_features = whitney_feature_selection(df)
# print(not_important_features)
# df.drop(not_important_features, axis=1, inplace=True)
#
# # Отбор лучших для классификации признаков
# save_the_most_important_features(df, 6)

# save_dataframe(df)
#
# scores = []
# # Делю датафрейм на тренировочный и тестовый
# for random_state in [0, 25, 50, 75, 100]:
#     X_train, X_test, y_train, y_test = split_dataframe(df, random_state)
#     scores.append(get_model_score(X_train, X_test, y_train, y_test))
# print(f'Cкор на тесте: {round(float(np.mean(scores)), 2)}')

# file1 = open('dataset/ЗД19a.txt', 'r')
# file2 = open('dataset/ЗД19b.txt', 'r')
# file3 = open('dataset/ЗД19c.txt', 'r')
#
# files = [file1, file2, file3]
# sample = get_sample_from_files(files)
#
# print(predict_class(sample))
