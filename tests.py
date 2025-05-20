import numpy as np
import pytest
from unittest import mock

from audio_processing import AudioProcessor, DefaultNoiseReduction, SimpleNoiseReduction
from feature_extraction import MFCCExtractor, SpectrogramExtractor, MelSpectrogramExtractor, FeatureExtractor
from model import EmotionRecognitionModel, ModelLogger
from sklearn.preprocessing import LabelEncoder


# ---------- Fixtures ----------
@pytest.fixture
def sample_audio():
    """Фікстура створює випадкове аудіо і sample rate"""
    return np.random.randn(22050), 22050


@pytest.fixture
def dummy_audio():
    """Фікстура створює випадковий одномірний сигнал"""
    return np.random.randn(22050)


# ---------- AudioProcessor Tests ----------

def test_load_audio_mono(tmp_path):
    """Перевіряє, що аудіо завантажується як моно і з вірною частотою дискретизації"""
    path = tmp_path / "test.wav"
    processor = AudioProcessor()
    processor.save_audio(path, np.random.randn(22050), 22050)
    audio, sr = processor.load_audio(str(path))
    assert audio.ndim == 1
    assert sr == 22050


def test_normalize_audio(sample_audio):
    """Перевіряє, що нормалізація масштабує сигнал до максимуму 1"""
    audio, _ = sample_audio
    processor = AudioProcessor()
    normalized = processor.normalize_audio(audio)
    assert np.isclose(np.max(np.abs(normalized)), 1.0, atol=1e-5)


def test_default_noise_reduction_called(sample_audio):
    """Перевіряє, що стратегія DefaultNoiseReduction викликає функцію noisereduce.reduce_noise"""
    with mock.patch("noisereduce.reduce_noise", return_value=sample_audio[0]) as mock_reduce:
        processor = AudioProcessor(noise_reduction_strategy=DefaultNoiseReduction())
        processor.reduce_noise(*sample_audio)
        mock_reduce.assert_called_once()


def test_simple_noise_reduction_does_nothing(sample_audio):
    """Перевіряє, що стратегія SimpleNoiseReduction не змінює сигнал"""
    processor = AudioProcessor(noise_reduction_strategy=SimpleNoiseReduction())
    result = processor.reduce_noise(*sample_audio)
    assert np.array_equal(result, sample_audio[0])


def test_normalize_zero_audio():
    """Перевіряє, що нормалізація нульового сигналу не дає NaN чи помилок"""
    processor = AudioProcessor()
    zero_audio = np.zeros(22050)
    normalized = processor.normalize_audio(zero_audio)
    assert np.all(normalized == 0)


# ---------- FeatureExtractor Tests ----------

def test_mfcc_shape(dummy_audio):
    """Перевіряє, що MFCC має 13 коефіцієнтів"""
    extractor = MFCCExtractor()
    mfcc = extractor.extract(dummy_audio)
    assert mfcc.shape[1] == 13


def test_spectrogram_output(dummy_audio):
    """Перевіряє, що спектрограма має 2 виміри"""
    extractor = SpectrogramExtractor()
    spectrogram = extractor.extract(dummy_audio)
    assert spectrogram.ndim == 2


def test_mel_spectrogram_output(dummy_audio):
    """Перевіряє, що mel-спектрограма має 2 виміри"""
    extractor = MelSpectrogramExtractor()
    mel = extractor.extract(dummy_audio)
    assert mel.ndim == 2


def test_feature_facade_extract_mfcc(dummy_audio):
    """Тестує фасад для MFCC"""
    facade = FeatureExtractor()
    mfcc = facade.extract_mfcc(dummy_audio)
    assert mfcc.shape[1] == 13


def test_feature_facade_extract_spectrogram(dummy_audio):
    """Тестує фасад для спектрограми"""
    facade = FeatureExtractor()
    spec = facade.extract_spectrogram(dummy_audio)
    assert spec.ndim == 2


# ---------- EmotionRecognitionModel Tests ----------

def test_model_build_shape():
    """Перевіряє, що модель має правильну форму входу"""
    model = EmotionRecognitionModel(input_shape=(13, 216, 1))
    assert model.model.input_shape == (None, 13, 216, 1)


def test_model_predict_shape():
    """Перевіряє, що результат передбачення має правильну форму"""
    model = EmotionRecognitionModel()
    dummy_input = np.random.rand(1, 13, 216, 1)
    preds = model.predict(dummy_input)
    assert preds.shape == (1,)


def test_model_train_does_not_crash():
    """Перевіряє, що тренування працює без помилок"""
    model = EmotionRecognitionModel()
    X = np.random.rand(10, 13, 216, 1)
    y = np.random.randint(0, 8, 10)
    model.train(X, y, X, y, epochs=1, batch_size=2)


def test_model_save_and_load(tmp_path):
    """Перевіряє збереження та завантаження моделі"""
    model_path = tmp_path / "model.h5"
    model = EmotionRecognitionModel()
    model.save_model(str(model_path))
    new_model = EmotionRecognitionModel()
    new_model.load_model(str(model_path))
    assert new_model.model is not None


def test_predict_invalid_input_shape():
    """Перевіряє, що передбачення з некоректним розміром викликає помилку"""
    model = EmotionRecognitionModel()
    with pytest.raises(Exception):
        model.predict(np.random.rand(13, 216))  # invalid shape


# ---------- Decorator Tests ----------

def test_model_logger_decorator_prints(capsys):
    """Перевіряє, що декоратор ModelLogger друкує час"""
    class DummyModel:
        def train(self, *args, **kwargs):
            return "trained"
        def predict(self, *args, **kwargs):
            return np.array([[0.1]*8])
    wrapped = ModelLogger(DummyModel())
    wrapped.train(None, None, None, None)
    wrapped.predict(np.random.rand(1, 13, 216, 1))
    captured = capsys.readouterr()
    assert "Training time" in captured.out
    assert "Prediction time" in captured.out


def test_logger_delegates_methods():
    """Перевіряє, що ModelLogger делегує методи обгортаному об'єкту"""
    class DummyModel:
        def some_method(self): return 42
    wrapper = ModelLogger(DummyModel())
    assert wrapper.some_method() == 42


# ---------- Utils / General ----------

def test_save_and_load_label_encoder(tmp_path):
    """Перевіряє збереження та завантаження енкодера міток"""
    le = LabelEncoder()
    le.fit(['happy', 'sad'])
    path = tmp_path / "labels.pkl"
    model = EmotionRecognitionModel()
    model.save_labels(str(path), le)
    loaded = model.load_labels(str(path))
    assert list(loaded.classes_) == ['happy', 'sad']


def test_imports():
    """Перевіряє, що всі необхідні імпорти працюють"""
    from audio_processing import AudioProcessor
    from feature_extraction import FeatureExtractor
    from model import EmotionRecognitionModel
    from utils import (
        load_audio_file, save_json, load_json, save_numpy_array,
        load_numpy_array, save_label_encoder, load_label_encoder, ensure_directory_exists
    )
    assert AudioProcessor is not None


def test_save_empty_audio_raises(tmp_path):
    """Перевіряє, що збереження порожнього сигналу не викликає помилку"""
    processor = AudioProcessor()
    path = tmp_path / "empty.wav"
    empty_audio = np.array([])
    try:
        processor.save_audio(path, empty_audio, 22050)
    except Exception as e:
        pytest.fail(f"Saving empty audio crashed with exception: {e}")
