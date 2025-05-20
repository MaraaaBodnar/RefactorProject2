import os
import numpy as np
import librosa
import joblib
import json


def load_audio_file(file_path, sample_rate=22050):
    """ Завантажує аудіофайл та повертає сигнал """
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio, sr


def save_json(data, file_path):
    """ Зберігає словник у JSON-файл """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    """ Завантажує словник із JSON-файлу """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_numpy_array(array, file_path):
    """ Зберігає масив NumPy у файл """
    np.save(file_path, array)


def load_numpy_array(file_path):
    """ Завантажує масив NumPy із файлу """
    return np.load(file_path)


def save_label_encoder(label_encoder, file_path):
    """ Зберігає енкодер міток """
    joblib.dump(label_encoder, file_path)


def load_label_encoder(file_path):
    """ Завантажує енкодер міток """
    return joblib.load(file_path)


def ensure_directory_exists(directory):
    """ Перевіряє, чи існує директорія, і створює її за потреби """
    if not os.path.exists(directory):
        os.makedirs(directory)