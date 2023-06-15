import os
import re
from typing import List, IO

from pandas import DataFrame
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from signal_preprocessing import get_signal_from_file, get_spectral_characteristics


def gen_dataset():
    for i, d, filenames in os.walk('dataset'):

        filenames = sorted(filenames, key=lambda filename: (
            len(re.split(r'[a-c]', filename)[0]), re.split(r'[a-c]', filename)[0]))

        dataset = dict()

        for filename in filenames:

            # Получаю сигнал из файла
            with open(f'dataset/{filename}', encoding='utf-8') as file:
                signal = get_signal_from_file(file)
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

        return dataset


def get_sample_from_files(files: List[IO]):

    chars = []
    for file in files:
        signal = get_signal_from_file(file)
        spectral_chars = get_spectral_characteristics(signal)
        chars += list(spectral_chars.values())

    column_names = ['SI-a', 'LF-a', 'HF-a', 'LF/HF-a', 'VLF-a', 'TP-a', 'SI-b', 'LF-b', 'HF-b', 'LF/HF-b',
                    'VLF-b', 'TP-b', 'SI-c', 'LF-c', 'HF-c', 'LF/HF-c', 'VLF-c', 'TP-c']
    df = DataFrame.from_dict({'sample': chars}, orient='index', columns=column_names)
    df = df[['HF-a', 'TP-a', 'SI-b', 'SI-c', 'LF-c', 'TP-c']]

    return df


def normalize_data_frame(df: DataFrame):
    # Создание объекта MinMaxScaler
    scaler = MinMaxScaler()

    # Применение MinMaxScaler к каждой колонке отдельно
    for column in df.columns:
        df[column] = scaler.fit_transform(df[[column]])


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
    label = df['cfs']
    df_without_labels = df.drop('cfs', axis=1, inplace=False)
    normalize_data_frame(df_without_labels)

    k_best = SelectKBest(k=k, score_func=chi2)
    k_best.fit(df_without_labels, label)
    k_best_features = k_best.get_feature_names_out()

    print(k_best_features)

    for column in df:
        if column not in k_best_features and column != 'cfs':
            df.drop(column, axis=1, inplace=True)


def split_dataframe(df: DataFrame, random_state: int):
    labels = df["cfs"]
    split_df = df.drop("cfs", axis=1, inplace=False)

    return train_test_split(
        split_df, labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels
    )


def save_dataframe(df: DataFrame):
    df.to_csv('dataframe.csv', index=False)

