import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),  ".."))
dir_store = os.path.join(os.path.dirname(__file__), "..", "data")
import time
import tensorflow as tf
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import math
from src.utils.tf_layer import set_input, add_layer
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=dir_store)


class GAN(object):

    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.verbose = config['verbose']
        self.noise_dim = config['noise_dim']
        self.discriminator_learning_rate = config['discriminator_learning_rate']
        self.generator_learning_rate = config['generator_learning_rate']
        self.beta1 = config['beta1']
        self.mini_batch_size = config['mini_batch_size']
        self.num_examples_to_generate = config['num_examples_to_generate']
        self.config = config

    def make_generator_network(self):
        prev_name = ""
        model = tf.keras.Sequential()
        model = set_input(model, {'input_shape': (self.noise_dim,)})
        for config_layer in self.config['generator_layers']:
            model = add_layer(model, config_layer, prev_name)
            prev_name = config_layer['name']
        print(model.summary())
        return model

    def make_discriminator_network(self):
        prev_name = ""
        model = tf.keras.Sequential()
        model = set_input(model, {'input_shape': self.input_shape})
        for config_layer in self.config['discriminator_layers']:
            model = add_layer(model, config_layer, prev_name)
            prev_name = config_layer['name']
        config_layer = {
            "name": "dense",
            "units": 1,
            "activation": "sigmoid"
        }
        model = add_layer(model, config_layer, prev_name)
        print(model.summary())
        return model

    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join(dir_store, os.path.join('gan_images', 'image_at_epoch_{}.png'.format(str(int(epoch))))))


    def random_mini_batches(self, X):
        mini_batches = []
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(self.m))
        shuffled_X = X[permutation, :, :]
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(self.m / self.mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :, :]
            mini_batch = (mini_batch_X)
            mini_batches.append(mini_batch)
        # Handling the end case (last mini-batch < mini_batch_size)
        if self.m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :, :]
            mini_batch = (mini_batch_X)
            mini_batches.append(mini_batch)
        return mini_batches

    def train(self, X):

        self.m, self.k, _, _ = X.shape
        self.input_shape = X.shape[1:]
        generator = self.make_generator_network()
        discriminator = self.make_discriminator_network()

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_learning_rate, beta_1=self.beta1)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_learning_rate,
                                                           beta_1=self.beta1)

        # @tf.function
        # def train_step(images)
        gen_loss_l = []
        real_loss_l = []
        disc_loss_l = []
        t = trange(self.num_epochs, desc="Epoch back-propagation", leave=True)
        for epoch in t:
            start = time.time()
            minibatches = self.random_mini_batches(X)
            for minibatch in minibatches:
                (minibatch_X) = minibatch
                noise = tf.random.normal([self.mini_batch_size, self.noise_dim])

                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                    generated_non_train = generator(noise, training=False)
                    real_output = discriminator(minibatch_X, training=True)
                    fake_output = discriminator(generated_non_train, training=True)

                    generated_train = generator(noise, training=True)

                    disc_gen_output = discriminator(generated_train, training=False)

                    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
                    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
                    disc_loss = real_loss + fake_loss

                    gen_loss = cross_entropy(tf.ones_like(disc_gen_output), disc_gen_output)

                    gen_loss_l.append(gen_loss.numpy())
                    real_loss_l.append(real_loss.numpy())
                    disc_loss_l.append(disc_loss.numpy())

                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

                discriminator_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, discriminator.trainable_variables))
                generator_optimizer.apply_gradients(
                    zip(gradients_of_generator, generator.trainable_variables))

                msg = "Disc Real Loss: {}. Disc Fake Loss: {}. Gen Loss: {}".format(
                    str(round(real_loss.numpy(), 4)),
                    str(round(fake_loss.numpy(), 4)),
                    str(round(gen_loss.numpy(), 4))
                )
                t.set_description(msg)
                t.refresh()

            # display.clear_output(wait=True)

            if self.num_examples_to_generate:
                self.generate_and_save_images(
                    generator,
                    epoch + 1,
                    tf.random.normal([self.num_examples_to_generate, self.noise_dim])
                )

            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # display.clear_output(wait=True)

        self.generator = generator
        return {
            "generator": generator,
            "stats": {
                "gen_loss_l": gen_loss_l,
                "disc_loss_l": disc_loss_l,
                "real_loss_l": real_loss_l
            }
        }
 
