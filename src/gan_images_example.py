import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
dir_store = os.path.join(os.path.dirname(__file__), "..", "data")
import tensorflow as tf
from src.gan_2D import GAN
import matplotlib.pyplot as plt


d = {
    "num_epochs": 100,
    "verbose": 1,
    "noise_dim": 100,
    "mini_batch_size": 256,
    "discriminator_learning_rate": 0.0001,
    "generator_learning_rate": 0.0005,
    "num_examples_to_generate": 16,
    "beta1": 0.5,
    "generator_layers": [
        {"name": "dense", "units": 7 * 7 * 256, "use_bias": True, "kernel_initializer": "he_uniform",
         "batch_normalization": True, "bias_initializer": "zeros", "activation": None, "reshape": (7, 7, 256)},
        {"name": "cnn2dT", "filters": 128, "kernel_size": (5, 5), "strides": (1, 1), "padding": "same", "use_bias": True,
        "batch_normalization": False,
         "kernel_initializer": "he_uniform", "bias_initializer": "zeros",
         "activation": "leaky_relu", "activation_config": {"alpha": 0.2}},
        {"name": "cnn2dT", "filters": 64, "kernel_size": (5, 5), "strides": (2, 2), "padding": "same",
         "use_bias": True, "kernel_initializer": "he_uniform", "bias_initializer": "zeros",
        "batch_normalization": False,
         "activation": "leaky_relu", "activation_config": {"alpha": 0.2}},
        {"name": "cnn2dT", "filters": 1, "kernel_size": (5, 5), "strides": (2, 2), "padding": "same",
         "use_bias": True, "kernel_initializer": "he_uniform", "bias_initializer": "zeros",
        "batch_normalization": False,
         "activation": "tanh"}
    ],
    "discriminator_layers": [
        {"name": "cnn2d", "filters": 64, "kernel_size": (5, 5), "strides": (2, 2), "padding": "same", "use_bias": True,
         "kernel_initializer": "glorot_uniform", "bias_initializer": "zeros","dropout": 0.3,
         "activation": "relu"},

        {"name": "cnn2d", "filters": 128, "kernel_size": (5, 5), "strides": (2, 2), "padding": "same", "use_bias": True,
         "kernel_initializer": "glorot_uniform", "bias_initializer": "zeros",
         "activation": "relu", "dropout": 0.3}
    ]
}


def get_default_data():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(train_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(dir_store, os.path.join('gan_images', 'image_at_real.png')))
    return train_images

def run():
    train_images = get_default_data()
    GAN(config=d).train(train_images)


if __name__ == '__main__':
    run()
