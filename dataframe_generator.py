import cmath
import os
import re
import statistics
import numpy as np
from scipy.stats import variation, mannwhitneyu
from hrvanalysis import remove_outliers, interpolate_nan_values, extract_features
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_spectral_characteristics(signal: list):
    # Секунды, на которых произошло сердечное сокращение
    time = signal

    # Получаю сигнал RR-интервалов
    signal = [round((signal[i + 1] - signal[i]) * 1000, ndigits=1) for i in range(len(time) - 1)]

    # Рассчитываю коэффициент вариации сигнала
    variation_coefficient = variation(signal)
    # Рассчитываю среднее значение сигнала
    mean = np.mean(signal)
    # Рассчитываю диапазон нормальной изменчивости значений сигнала
    nn_diapason = variation_coefficient * 3 * 1000

    # Получаю секунды, на которых был найден артефакт
    seconds = []
    for i in range(len(signal)):
        if signal[i] < mean - nn_diapason or signal[i] > mean + nn_diapason:
            seconds.append(signal[i])
    # print(f'секунды выбросов: {seconds}')

    # Чистка сигнала от артефактов
    signal_without_artefacts = remove_outliers(signal, low_rri=mean - nn_diapason, high_rri=mean + nn_diapason,
                                               verbose=False)
    # Интерполяция сигнала методом кубического сплайна (получение сигнала, состоящего из NN-интервалов)
    interpolated_signal = interpolate_nan_values(signal_without_artefacts, interpolation_method='cubicspline')
    # Извлечение спектральных характеристик из интерполированного сигнала
    spectral_features = extract_features.get_frequency_domain_features(nn_intervals=interpolated_signal, method='welch',
                                                                       interpolation_method='cubic')
    # Расчет стресс-индекса (SI) по интерполированному сигналу
    stress_index = get_stress_index_from_signal(interpolated_signal)
    spectral_features['SI'] = stress_index

    # Переименование полученных спектральных характеристик в читабельный вид
    rename_dict_keys_names(spectral_features)

    # Возвращаю вычисленные спектральные характеристики вместе со стресс-индексом
    return spectral_features


def get_stress_index_from_signal(signal: list):
    # Мода сигнала
    mode = statistics.mode(signal) / 1000
    # Амплитуда моды сигнала
    mode_amplitude = signal.count(mode * 1000) / len(signal) * 100
    # Разница между наибольшим и наименьшим интервалом сигнала
    Mx_DMn = (max(signal) - min(signal)) / 1000

    # Расчет стресс-индекса по формуле Баевского
    stress_index = mode_amplitude / (2 * mode * Mx_DMn)
    return stress_index


def rename_dict_keys_names(dic: dict):
    # Удаляю ненужные спектральные характеристики
    dic.pop('lfnu')
    dic.pop('hfnu')

    # Перевожу ключи словаря в читабельный вид
    dic['LF'] = dic.pop('lf')
    dic['HF'] = dic.pop('hf')
    dic['LF/HF'] = dic.pop('lf_hf_ratio')
    dic['VLF'] = dic.pop('vlf')
    dic['TP'] = dic.pop('total_power')


def get_signal_from_file(filename):
    with open(f'dataset/{filename}', encoding='utf-8') as file:
        signal = []
        lines = file.readlines()
        for line in lines:
            split = line.split()
            signal.append(float(split[0]))

        return signal


def gen_dataset():
    for i, d, filenames in os.walk('dataset'):
        dataset = dict()
        for filename in filenames:

            # Получаю сигнал из файла
            signal = get_signal_from_file(filename)
            # Получаю спектральные характеристики сигнала
            spectral_characteristics = get_spectral_characteristics(signal)

            split = re.split(r'[a-c]', filename)

            # Получаю идентификатор пациента
            patient = split[0]

            # Присваиваю ему спектральные характеристики, рассчитанные в разных пробах
            if patient in dataset:
                dataset[patient] += list(spectral_characteristics.values())
            else:
                if patient.startswith("ЗД"):
                    label = 0
                else:
                    label = 1
                dataset[patient] = [label] + list(spectral_characteristics.values())

        dataset = dict(sorted(dataset.items(), key=lambda patient: (len(patient[0]), patient[0])))
        return dataset


def get_dataframe(dataset: dict):
    column_names = ['cfs', 'SI-a', 'LF-a', 'HF-a', 'LF/HF-a', 'VLF-a', 'TP-a', 'SI-b', 'LF-b', 'HF-b', 'LF/HF-b',
                    'VLF-b',
                    'TP-b', 'SI-c', 'LF-c', 'HF-c', 'LF/HF-c', 'VLF-c', 'TP-c']
    df = DataFrame.from_dict(dataset, orient='index', columns=column_names)
    return df


def normalize_data_frame(df: DataFrame):
    # Создание объекта MinMaxScaler
    scaler = MinMaxScaler()

    # Применение MinMaxScaler к каждой колонке отдельно
    for column in df.columns:
        df[column] = scaler.fit_transform(df[[column]])


def homogen_by_wilcoxon(x: list, y: list):
    m = len(x)
    n = len(y)

    # Если длина первой выборки больше длины второй, то меняем их местами
    if m > n:
        z = x
        x = y
        y = z

    x = [[val, 1] for val in x]
    y = [[val, 2] for val in y]

    d = x + y
    d = sorted(d, key=lambda el: el[0])
    for i in range(len(d)):
        d[i].append(i + 1)

    same = dict()
    for el in d:
        if same.__contains__(el[0]):
            same[el[0]].append(el[2])
        else:
            same[el[0]] = [el[2]]

    for el in d:
        values = same[el[0]]
        el[2] = np.mean(values)

    x_ranks = []
    for el in d:
        if el[1] == 1:
            x_ranks.append(el[2])
    w = sum(x_ranks)

    INVERSE_FUNCTION_VALUE = 1.959964

    w_left = abs((m * (m + n + 1) - 1) / 2 - INVERSE_FUNCTION_VALUE * cmath.sqrt(m * n * (m + n + 1) / 12))
    w_right = m * (m + n + 1) - w_left

    return w_left <= w <= w_right


def whitney_feature_selection(df: DataFrame):
    # Разбиваем датафрейм на 2 выборки: больные и здоровые:
    cfs_df = df[df['cfs'] == 1]
    healthy = df[df['cfs'] == 0]

    not_important_features = []

    for column in df:

        if column != 'cfs':

            cfs_feature_values = cfs_df[column].values
            healthy_feature_values = healthy[column].values

            stat, p = mannwhitneyu(healthy_feature_values, cfs_feature_values)
            if p > 0.05:
                not_important_features.append(column)

    return not_important_features


def save_the_most_important_features(df: DataFrame, k: int):
    label = df["cfs"]
    df_without_labels = df.drop("cfs", axis=1, inplace=False)
    normalize_data_frame(df_without_labels)

    k_best = SelectKBest(k=k, score_func=chi2)
    k_best.fit_transform(df_without_labels, label)
    k_best_features = k_best.get_feature_names_out()

    for column in df:
        if column not in k_best_features and column != 'cfs':
            df.drop(column, axis=1, inplace=True)


def split_dataframe(df: DataFrame):
    labels = df["cfs"]
    df.drop("cfs", axis=1, inplace=True)

    return train_test_split(
        df, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )


def use_knn(X_train, X_test, y_train, y_test):

    clf = KNeighborsClassifier(n_neighbors=3, leaf_size=1, weights='distance')
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('knn', clf)])

    loo = LeaveOneOut()

    scores = []
    for train_index, test_index in loo.split(X_train, y_train):

        fold_X_train, fold_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        pipeline.fit(fold_X_train, fold_y_train)

        y_pred = pipeline.predict(fold_X_test)
        score = accuracy_score(fold_y_test, y_pred)

        scores.append(score)

    print(f'Средний скор модели на кросс-валидации: {round(float(np.mean(scores)), 2)}')
    print(f'Cкор на тесте: {round(pipeline.score(X_test, y_test), 2)}')

    # scores = cross_val_score(pipeline, X_train, y_train, cv=loo, scoring='accuracy')
    # pipeline.fit(X_train, y_train)
    #
    # score_in_test = pipeline.score(X_test, y_test)
    # print(f'Скор на тесте: {round(score_in_test, 2)}')

    # knn_params = {'knn__n_neighbors': list(range(1, 20, 2)),
    #               'knn__weights': ['uniform', 'distance'],
    #               'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #               'knn__leaf_size': list(range(1, 10)),
    #               'knn__metric': ['minkowski', 'cityblock', 'euclidean', 'l1', 'l2', 'manhattan']}
    #
    # grid_pipeline = GridSearchCV(estimator=pipeline,
    #                              param_grid=knn_params,
    #                              cv=loo,
    #                              scoring='accuracy',
    #                              return_train_score=True,
    #                              verbose=2)
    # grid_pipeline.fit(X_train, y_train)
    #
    # print(grid_pipeline.best_params_)


def use_svm(X_train, X_test, y_train, y_test):

    clf = SVC(C=4, kernel='linear', probability=True, random_state=42)
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('svm', clf)])

    loo = LeaveOneOut()

    scores = []
    for train_index, test_index in loo.split(X_train, y_train):

        fold_X_train, fold_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        pipeline.fit(fold_X_train, fold_y_train)

        y_pred = pipeline.predict(fold_X_test)
        score = accuracy_score(fold_y_test, y_pred)

        scores.append(score)

    print(f'Средний скор модели на кросс-валидации: {round(float(np.mean(scores)), 2)}')
    print(f'Cкор на тесте: {round(pipeline.score(X_test, y_test), 2)}')

    # svm_params = {'svm__C': [0.100, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #               'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #               'svm__gamma': ['scale', 'auto'],
    #               'svm__shrinking': [True, False],
    #               'svm__probability': [True, False]}
    #
    # grid_pipeline = GridSearchCV(estimator=pipeline,
    #                              param_grid=svm_params,
    #                              cv=loo,
    #                              scoring='accuracy',
    #                              verbose=2)
    #
    # grid_pipeline.fit(X_train, y_train)
    # print(grid_pipeline.best_params_)


# Генерация датасета
dataset = gen_dataset()
# Генерация датафрейма из датасета
df = get_dataframe(dataset)

# Удаление признаков, однородных по выборкам
not_important_features = whitney_feature_selection(df)
df.drop(not_important_features, axis=1, inplace=True)

# Отбор лучших для классификации признаков
save_the_most_important_features(df, 15)

# Делю датафрейм на тренировочный и тестовый
X_train, X_test, y_train, y_test = split_dataframe(df)

use_svm(X_train, X_test, y_train, y_test)
