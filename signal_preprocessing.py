import statistics

import numpy as np
from hrvanalysis import interpolate_nan_values, extract_features


def get_signal_from_file(file):
    signal = []
    lines = file.readlines()
    for line in lines:
        split = line.split()
        signal.append(float(split[0]))

    return signal


def get_stress_index_from_signal(signal: list):

    # Мода сигнала
    mode = statistics.mode(signal)
    # Амплитуда моды сигнала
    mode_amplitude = signal.count(mode) / len(signal) * 100
    # Разница между наибольшим и наименьшим интервалом сигнала
    Mx_DMn = (max(signal) - min(signal)) / 1000

    # Расчет стресс-индекса по формуле Баевского
    stress_index = mode_amplitude / (2 * (mode / 1000) * Mx_DMn)
    return stress_index


def get_spectral_characteristics(signal: list):

    # Получаю сигнал RR-интервалов
    rr_intervals = [round((signal[i + 1] - signal[i]) * 1000, ndigits=1) for i in range(len(signal) - 1)]

    remove_artefacts(rr_intervals)

    # Интерполяция удаленных значений в сигнале методом кубического сплайна
    interpolated_signal = interpolate_nan_values(rr_intervals, interpolation_method='cubicspline')

    # Извлечение спектральных характеристик из сигнала NN-интервалов
    spectral_features = extract_features.get_frequency_domain_features(nn_intervals=interpolated_signal,
                                                                       interpolation_method='cubic')
    # Расчет стресс-индекса (SI) по интерполированному сигналу
    stress_index = get_stress_index_from_signal(interpolated_signal)
    spectral_features['SI'] = stress_index

    # Переименование полученных спектральных характеристик в читабельный вид
    rename_dict_keys_names(spectral_features)

    # Возвращаю вычисленные спектральные характеристики вместе со стресс-индексом
    return spectral_features


def remove_artefacts(rr_intervals: list):

    for i in range(1, len(rr_intervals) - 1):

        prev = rr_intervals[i - 1]
        curr = rr_intervals[i]
        next = rr_intervals[i + 1]

        if next >= 2 * prev or (curr / prev < 0.755 and next > prev):
            rr_intervals[i] = np.nan
            rr_intervals[i + 1] = np.nan
            i += 2


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

