import tensorflow as tf


class SimpleAutoencoder:
    def __init__(self, input_shape: (int, int, int) = (28, 28, 1)):
        input_tensor = tf.keras.layers.Input(shape=input_shape)

        # encoder
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

        # decoder
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

        self.__model = tf.keras.Model(input_tensor, x)

    @property
    def model(self):
        return self.__model

    def print_summary(self):
        return self.__model.summary()
