import tensorflow as tf


class SimpleVAE:
    def __init__(self,
                 input_dim: int = 28 * 28,
                 intermediate_layer_dim: int = 64,
                 latent_dim: int = 2):
        self.__latent_dim = latent_dim
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        h = tf.keras.layers.Dense(intermediate_layer_dim, activation='relu')(inputs)
        self.__z_mean = tf.keras.layers.Dense(latent_dim)(h)
        self.__z_log_sigma = tf.keras.layers.Dense(latent_dim)(h)
        self.__z = tf.keras.layers.Lambda(self.sample)((self.__z_mean, self.__z_log_sigma))

        # encoder
        self.__encoder = tf.keras.Model(inputs, [self.__z_mean, self.__z_log_sigma, self.__z], name='encoder')

        # decoder
        latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(intermediate_layer_dim, activation='relu')(latent_inputs)
        outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
        self.__decoder = tf.keras.Model(latent_inputs, outputs)

        # VAE model
        outputs = self.__decoder(self.__encoder(inputs)[2])
        self.__vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

        # custom loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dim
        kl_loss = 1 + self.__z_log_sigma - tf.keras.backend.square(self.__z_mean) - tf.exp(self.__z_log_sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.__vae.add_loss(vae_loss)

    def sample(self, args: (tf.Tensor, tf.Tensor)) -> tf.Tensor:
        mean, log_sigma = args
        epsilon = tf.keras.backend.random_normal(
            shape=(tf.shape(mean)[0], self.__latent_dim),
            mean=0.,
            stddev=0.1
        )
        return mean + tf.exp(log_sigma) * epsilon

    @property
    def encoder(self) -> tf.keras.Model:
        return self.__encoder

    @property
    def decoder(self) -> tf.keras.Model:
        return self.__decoder

    @property
    def vae(self) -> tf.keras.Model:
        return self.__vae
