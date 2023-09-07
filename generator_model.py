from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LeakyReLU,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf

# Load MNIST data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = x_train.reshape(x_train.shape[0], 784)  # Reshape to 784-dimensional vectors

# One-hot encoding the labels
y_train = to_categorical(y_train, num_classes=10)

# Hyperparameters
batch_size = 128
epochs = 1000
latent_dim = 100
sample_interval = 100


# Generator
def build_generator():
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(10,))  # Changed from (1,)
    label_embedding = Dense(latent_dim)(label)

    merged = Concatenate(axis=-1)([noise, label_embedding])
    x = Dense(256)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(784, activation="tanh")(x)

    generator = Model([noise, label], x)
    return generator


# Discriminator
def build_discriminator():
    img = Input(shape=(784,))
    label = Input(shape=(10,))  # Changed from (1,)
    label_embedding = Dense(784)(label)

    merged = Concatenate(axis=-1)([img, label_embedding])
    x = Dense(512)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation="sigmoid")(x)

    discriminator = Model([img, label], x)
    return discriminator


# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(
    loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"]
)

# Build the generator
generator = build_generator()

# For the combined model we will only train the generator
noise = Input(shape=(latent_dim,))
label = Input(shape=(10,))  # Changed from (1,)
img = generator([noise, label])
discriminator.trainable = False
valid = discriminator([img, label])
combined = Model([noise, label], valid)
combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

# Train the GAN
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs, labels = x_train[idx], y_train[idx]  # labels are now one-hot encoded

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict([noise, labels])

    d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    sampled_labels = to_categorical(
        np.random.randint(0, 10, (batch_size, 1)), num_classes=10
    )

    g_loss = combined.train_on_batch([noise, sampled_labels], valid)

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

# Save the trained generator model
save_model(generator, "generator_model.h5")
