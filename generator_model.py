from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LeakyReLU,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load in the data and preprocess
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(x_train.shape[0], 784)
y_train = y_train.reshape(-1, 1)

# Hyperparameters
batch_size = 128
epochs = 10000
sample_interval = 1000
latent_dim = 100
num_classes = 10


# Build the generator model
def build_generator():
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    label_embedding = Dense(latent_dim)(label)

    merged_input = Concatenate()([noise, label_embedding])

    x = Dense(256)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(784, activation="tanh")(x)

    model = Model([noise, label], x)
    return model


# Build the discriminator model
def build_discriminator():
    img = Input(shape=(784,))
    label = Input(shape=(1,))
    label_embedding = Dense(784)(label)
    merged_input = Concatenate()([img, label_embedding])

    x = Dense(512)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model([img, label], x)
    return model


# Compile models
discriminator = build_discriminator()
discriminator.compile(
    optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"]
)

generator = build_generator()

noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])

discriminator.trainable = False
validity = discriminator([img, label])

combined = Model([noise, label], validity)
combined.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")

# Training
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs, labels = x_train[idx], y_train[idx]

    noise = np.random.randn(batch_size, latent_dim)
    gen_imgs = generator.predict([noise, labels])

    d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.randn(batch_size, latent_dim)
    sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

    g_loss = combined.train_on_batch([noise, sampled_labels], valid)

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

# Save the generator model
generator.save("generator_model.h5")
