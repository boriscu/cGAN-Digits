from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained generator model
generator = load_model("generator_model.h5")

# Initialize plot
fig, ax = plt.subplots()
plt.axis("off")

while True:
    # Read a digit from the user
    digit = input("Enter a digit (or 'q' to quit): ")

    if digit == "q":
        break

    if not digit.isdigit() or not (0 <= int(digit) <= 9):
        print("Please enter a valid digit.")
        continue

    # Generate the random noise vector (latent space)
    noise = np.random.randn(1, 100)

    # Create the label for the digit (one-hot encoded)
    label = np.zeros((1, 10))
    label[0, int(digit)] = 1

    # Generate the image
    img = generator.predict([noise, label])

    # Post-process the image
    img = (img + 1) / 2.0

    # Update the existing plot with new image data
    ax.imshow(img[0, :, :, 0], cmap="gray")
    plt.draw()
    plt.pause(0.001)
