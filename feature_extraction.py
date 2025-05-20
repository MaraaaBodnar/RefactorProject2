import librosa
import numpy as np
from abc import ABC, abstractmethod


# 2. Шаблон проектування Factory для різних ознак
class FeatureExtractorFactory(ABC):
    @abstractmethod
    def extract(self, audio):
        pass


# Конкретні фабрики для кожного типу ознак
class MFCCExtractor(FeatureExtractorFactory):
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc.T


class SpectrogramExtractor(FeatureExtractorFactory):
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio):
        spectrogram = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        return np.abs(spectrogram)


class MelSpectrogramExtractor(FeatureExtractorFactory):
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return librosa.power_to_db(mel_spectrogram, ref=np.max)


class FeatureExtractor:
    """Фасад для зворотної сумісності. Використовує фабричні методи всередині."""
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        self.mfcc_extractor = MFCCExtractor(sample_rate, n_mfcc, n_fft, hop_length)
        self.spectrogram_extractor = SpectrogramExtractor(n_fft, hop_length)
        self.mel_spectrogram_extractor = MelSpectrogramExtractor(sample_rate, n_fft, hop_length)

    def extract_mfcc(self, audio):
        return self.mfcc_extractor.extract(audio)

    def extract_spectrogram(self, audio):
        return self.spectrogram_extractor.extract(audio)

    def extract_mel_spectrogram(self, audio):
        return self.mel_spectrogram_extractor.extract(audio)
