import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from abc import ABC, abstractmethod


# 1. Шаблон проектування Strategy для шумозаглушення
class NoiseReductionStrategy(ABC):
    @abstractmethod
    def reduce_noise(self, audio, sr):
        pass


class DefaultNoiseReduction(NoiseReductionStrategy):
    """Конкретна стратегія (використовує noisereduce)"""
    def reduce_noise(self, audio, sr):
        return nr.reduce_noise(y=audio, sr=sr)


class SimpleNoiseReduction(NoiseReductionStrategy):
    """Альтернативна стратегія (простий фільтр, без реального шумозаглушення)"""
    def reduce_noise(self, audio, sr):
        return audio


class AudioProcessor:
    """Клас для обробки аудіофайлів: завантаження, нормалізація, шумозаглушення."""
    def __init__(
        self,
        sample_rate=22050,
        noise_reduction_strategy=DefaultNoiseReduction()  # Стратегія за замовчуванням
    ):
        self.sample_rate = sample_rate
        self.noise_reduction = noise_reduction_strategy  # Ін'єкція залежності

    def load_audio(self, file_path):
        """Завантажує аудіофайл та конвертує його у моно"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True, res_type="kaiser_fast")
        return audio, sr

    @staticmethod
    def normalize_audio(audio):
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return np.zeros_like(audio)
        return audio / max_val

    def reduce_noise(self, audio, sr):
        """Виконує шумозаглушення через обрану стратегію"""
        return self.noise_reduction.reduce_noise(audio, sr)

    @staticmethod
    def save_audio(file_path, audio, sr):
        sf.write(str(file_path), audio, sr)
