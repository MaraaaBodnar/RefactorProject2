import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
import time


# 3. Шаблон проектування Decorator для логування
class ModelLogger:
    """Додає логування часу виконання методів train та predict"""
    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train, x_val, y_val, *args, **kwargs):
        start_time = time.time()
        result = self.model.train(x_train, y_train, x_val, y_val, *args, **kwargs)
        print(f"Training time: {time.time() - start_time:.2f}s")
        return result

    def predict(self, x, *args, **kwargs):
        start_time = time.time()
        result = self.model.predict(x, *args, **kwargs)
        print(f"Prediction time: {time.time() - start_time:.2f}s")
        return result

    def __getattr__(self, name):
        return getattr(self.model, name)


class EmotionRecognitionModel:
    """Клас для створення, навчання, збереження та передбачення емоційної моделі."""

    def __init__(self, input_shape=(13, 216, 1), num_classes=8, enable_logging=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

        if enable_logging:
            self.model = ModelLogger(self.model)

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
        """ Навчає модель на заданих даних """
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

    def predict(self, x):
        """ Робить передбачення на нових зразках """
        predictions = self.model.predict(x)
        return np.argmax(predictions, axis=1)

    def save_model(self, model_path):
        """ Зберігає модель у файл """
        self.model.save(model_path)

    def load_model(self, model_path):
        """ Завантажує модель з файлу """
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def save_labels(label_path, label_encoder):
        """ Зберігає енкодер міток """
        joblib.dump(label_encoder, label_path)

    @staticmethod
    def load_labels(label_path):
        """ Завантажує енкодер міток """
        return joblib.load(label_path)
