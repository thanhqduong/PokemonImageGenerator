import os
import random

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
import tensorflow as tf

from tensorflow.keras.models import load_model
gan_model = load_model('gen.h5')
def generate_gray():
  noise = np.random.normal(0,1, size = [1, 2000])
  flat = gan_model.predict(noise)
  small_img = flat.reshape(64, 64)
  img = cv.resize(small_img, (256, 256), interpolation = cv.INTER_AREA)
  if random.random() > 0.5:
    img = flip_image(img)
  return img 

  def flip_image(img):
      l = img.tolist()
  ret = []
  for r in l:
    ret.append(r[::-1])
  return np.array(ret)

img_rows, img_cols = 256, 256
nb_neighbors = 5
T = 0.38
colorize_model = load_model('color.h5')
epsilon = 1e-8

h, w = img_rows // 4, img_cols // 4


q_ab = np.load("pts_in_hull.npy")
nb_q = q_ab.shape[0]

nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

def generate_image(nums):
    for i in range(nums):

        gray = generate_gray()
        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        X_colorized = colorize_model.predict(x_test)
        X_colorized = X_colorized.reshape((h * w, nb_q))

        X_colorized = np.exp(np.log(X_colorized) / T) #
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        q_a = q_ab[:, 0].reshape((1, 313))
        q_b = q_ab[:, 1].reshape((1, 313))

        X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
        X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))

        X_a = cv.resize(X_a, (img_rows, img_cols), cv.INTER_CUBIC)
        X_b = cv.resize(X_b, (img_rows, img_cols), cv.INTER_CUBIC)


        X_a = X_a + 128
        X_b = X_b + 128


        out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
        out_lab[:, :, 0] = gray* 255
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b

        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv.cvtColor(out_lab, cv.COLOR_LAB2BGR)

        out_bgr = out_bgr.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')


        f, axarr = plt.subplots(1,1)
        axarr[0].imshow(out_bgr)

generate_image(1)