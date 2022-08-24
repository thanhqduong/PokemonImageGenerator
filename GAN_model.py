import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os


np.random.seed(1000)
random_dim = 2000

def load_data():
    x_train = pd.read_csv('data.csv', header = None)
    x_train = x_train.to_numpy()
    return x_train

def get_optimizer():
  return Adam(learning_rate=0.0002, beta_1=0.5)

def get_generator(optimizer, ip):
  generator = Sequential()

  generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
  generator.add(LeakyReLU(0.2)) 

  generator.add(Dense(1024))
  generator.add(LeakyReLU(0.2))

  generator.add(Dense(2048))
  generator.add(LeakyReLU(0.2))
  
  generator.add(Dense(ip, activation='tanh'))
  generator.compile(loss='binary_crossentropy', optimizer=optimizer)
  return generator


def get_discriminator(optimizer, ip):
  discriminator = Sequential()

  discriminator.add(Dense(2048, input_dim=ip, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))

  discriminator.add(Dense(1024))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))

  discriminator.add(Dense(512))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))

  discriminator.add(Dense(256))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))
  
  discriminator.add(Dense(1, activation='sigmoid'))
  discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
  return discriminator



def get_gan_network(discriminator, random_dim, generator, optimizer):
    
  discriminator.trainable = False
  gan_input = Input(shape=(random_dim,))
  x = generator(gan_input)
  gan_output = discriminator(x)
  gan = Model(inputs=gan_input, outputs=gan_output)
  gan.compile(loss='binary_crossentropy', optimizer=optimizer)
  return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize= (10,10)):
      
  noise = np.random.normal(0, 1, size=[examples, random_dim])
  generated_images = generator.predict(noise)
  generated_images = generated_images.reshape(examples, 64, 64)
  plt.figure(figsize=figsize)
  for i in range(generated_images.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
    plt.axis('off')
  plt.tight_layout()
  plt.savefig('generate/image_epoch_%d.png' % epoch)


def train(epochs=1, batch_size=128):
    
  k = 64*64
  x_train = load_data()

  batch_count = x_train.shape[0] / batch_size

  adam = get_optimizer()
  generator = get_generator(adam, k)
  discriminator = get_discriminator(adam, k)
  gan = get_gan_network(discriminator, random_dim, generator, adam)
  for e in range(1, epochs+1):
    print('-'*15, 'Epoch %d' % e, '-'*15)
    for _ in tqdm(range(int(batch_count))):

      noise = np.random.normal(0, 1, size=[batch_size, random_dim])
      image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

      generated_images = generator.predict(noise)
      X = np.concatenate([image_batch, generated_images])
      y_dis = np.zeros(2*batch_size)

      y_dis[:batch_size] = 0.9

      discriminator.trainable = True
      discriminator.train_on_batch(X, y_dis)

      noise = np.random.normal(0, 1, size=[batch_size, random_dim])
      y_gen = np.ones(batch_size)
      discriminator.trainable = False
      gan.train_on_batch(noise, y_gen)
    if e == 1 or e % 50 == 0:
      plot_generated_images(e, generator)
  generator.save('gen.h5')


train(6500, 32)