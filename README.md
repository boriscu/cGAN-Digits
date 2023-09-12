# cGan for Handwritten Digits Generation

## Introduction

This repository contains a Conditional Generative Adversarial Network (cGAN) that generates handwritten digits. The cGAN network consists of a Generator and a Discriminator, both implemented using TensorFlow's Keras API.

## Features

- **Data Set**: Uses the MNIST data set for handwritten digits.

- **Discriminator**: Uses Leaky ReLU activation, Binary Cross-Entropy loss, and Adam optimizer.

- **Generator**: Utilizes dense layers, batch normalization, and Leaky ReLU activation functions.

- **Hyperparameters**:

- Batch size \(= 128\)

- Epochs \(= 10000\)

- Latent dimensions \(= 100\)

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Code Structure

GAN's can be explained as the contest between a forger and a detective. In this narrative, the forger's goal is to create counterfeit banknotes so convincing that they are indistinguishable from real ones. The detective's mission, on the other hand, is to identify these fakes from the real deal.

In the GAN architecture, the generator plays the role of the forger, while the discriminator acts as the detective. Both engage in an iterative learning process: the generator strives to produce more convincing "fakes," while the discriminator endeavors to get better at distinguishing them from real data. Over time, the generator gets so good at its task that the discriminator finds it increasingly difficult to tell real from fake, leading to a well-trained GAN.

### Building the Generator

In our analogy, the generator or the "forger" starts with random 'noise' and a 'label' (representing the digit to generate) as inputs.

```python
noise = Input(shape=(latent_dim,))
label = Input(shape=(10,))  # One-hot encoded label
```

The label is transformed into a dense layer (`label_embedding`) that has the same dimensions as the noise, so that they can be effectively concatenated.

```python
label_embedding = Dense(latent_dim)(label)
merged = Concatenate(axis=-1)([noise, label_embedding])
```

The concatenated input then passes through several dense layers, leaky ReLU activation functions, and batch normalization layers to finally produce a 784-dimensional output that represents a flattened 28x28 image of a handwritten digit.

```python
x =  Dense(256)(merged)
x =  LeakyReLU(alpha=0.2)(x)
x =  BatchNormalization()(x)
x =  Dense(512)(x)
x =  LeakyReLU(alpha=0.2)(x)
x =  BatchNormalization()(x)
x =  Dense(1024)(x)
x =  LeakyReLU(alpha=0.2)(x)
x =  BatchNormalization()(x)
x =  Dense(784,  activation="tanh")(x)
```

#### The Role of Latent Space

Latent space acts as a sort of 'idea incubator' for the generator. By navigating through different points in this space, the generator learns to produce more convincing forgeries. Essentially, the generator is on a quest to find those regions in the latent space that can trick the discriminator most effectively.
Latent space can be thought of as a hidden realm from which our forger (generator) draws inspiration to create new banknotes (images). This 'realm' is filled with random numerical values that serve as the building blocks for these creations
Here's where the training loop brings the latent space into play:

```python
noise = np.random.normal(0, 1, (batch_size, latent_dim))
gen_imgs = generator.predict([noise, labels])
```

During each epoch of the training, new random points are sampled from the latent space. These points, when fed into the generator, result in new generated images that are then used to train both the discriminator and, indirectly, the generator via the combined model.

### Building the Discriminator

In our analogy, the discriminator or the "detective" receives an 'image' and a 'label' as inputs, where the image could either be real or generated by the forger (generator).

```python
img = Input(shape=(784,))
label = Input(shape=(10,))  # One-hot encoded label
```

Just like in the generator, the label is transformed into a dense layer (`label_embedding`) that has the same dimensions as the image. These are then concatenated.

```python
label_embedding = Dense(784)(label)
merged = Concatenate(axis=-1)([img, label_embedding])
```

This merged input is fed through dense layers and leaky ReLU activation functions to produce a single scalar output between 0 and 1, representing the discriminator's confidence that the image is real.

```python
x =  Dense(512)(merged)
x =  LeakyReLU(alpha=0.2)(x)
x =  Dense(256)(x)
x =  LeakyReLU(alpha=0.2)(x)
x =  Dense(1,  activation="sigmoid")(x)
```

The discriminator and generator are then compiled and trained against each other in a loop, each improving through the contest, thus completing our analogy. With each iteration, the generator becomes a better "forger," while the discriminator becomes a sharper "detective."

### Combined Model

After individually creating the generator and discriminator, we set up a combined model. In the context of our analogy, think of the combined model as a courtroom scenario where the forger (generator) is put to test. The "judge" (discriminator) has to decide if the produced banknote (image) is real or fake based on the evidence presented.

```python
noise = Input(shape=(latent_dim,))
label = Input(shape=(10,))
img = generator([noise, label])
valid = discriminator([img, label])
combined = Model([noise, label], valid)
combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))
```

In this combined model, random noise and a label are fed into the generator to produce an image. This generated image, along with its label, is then passed to the discriminator. However, unlike before, the goal is not to fool the discriminator but to adjust the generator in such a way that the discriminator thinks the generated image is real. Essentially, the combined model focuses on training the generator to improve its forgery skills, assuming that the discriminator is an expert judge.

### Freezing the Discriminator

Before training this combined model, it's essential to 'freeze' the discriminator. This is akin to telling our detective or "judge" to suspend their judgment while the forger learns to improve. Essentially, we want to adjust the generator's parameters without affecting the discriminator.

```python
discriminator.trainable = False
```

By setting `discriminator.trainable = False`, we ensure that the discriminator's weights are not updated during the training of the combined model. This is crucial because, in the training loop, we alternate between training the discriminator and the generator. The discriminator is updated based on both real and generated images, while the generator is updated based on the combined model.

### Training Loop

The training loop involves alternating between these steps:

1.  Train the discriminator on both real and generated data (weights can change).
2.  Freeze the discriminator, train the generator via the combined model (only the generator's weights can change).

```python
for epoch in  range(epochs):
	idx = np.random.randint(0, x_train.shape[0], batch_size)
	imgs, labels = x_train[idx], y_train[idx]
	noise = np.random.normal(0,  1,  (batch_size, latent_dim))
	gen_imgs = generator.predict([noise, labels])
	d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
	d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
	d_loss =  0.5  * np.add(d_loss_real, d_loss_fake)
	noise = np.random.normal(0,  1,  (batch_size, latent_dim))
	sampled_labels =  to_categorical(np.random.randint(0,  10,  (batch_size,  1)),  num_classes=10)
	g_loss = combined.train_on_batch([noise, sampled_labels], valid)
```

This interleaved training regimen ensures that neither the generator nor the discriminator becomes too powerful, maintaining a dynamic equilibrium where both are continually improving—akin to an ongoing duel between a forger and a detective, each honing their skills over time.
## Training Journey
The following sequence of images showcases the evolution of GAN. As the epochs progress, observe the remarkable improvement in the quality of the generated numbers.

![](https://github.com/boriscu/cGAN-Digits/blob/main/public/Images_Through_Epochs.png)

After diving deep into the training process of our GAN model, it's time to put its craftsmanship to the ultimate test: a side-by-side comparison with real handwritten digits. The following set of images presents a compelling contrast between genuine samples from the MNIST dataset and their artificially generated counterparts.

![](https://github.com/boriscu/cGAN-Digits/blob/main/public/realVsFake.png)
