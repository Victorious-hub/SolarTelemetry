import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from dataset import get_temperature

def draw_record(record, dec_record, show=True,  filename='', save=False):
  index = []
  temperature = []
  voltage = []
  current = []
  irradiating = []
  for r in record:
    index.append(float(r[0]))
    temperature.append(float(r[1]))
    voltage.append(float(r[2]))
    current.append(float(r[3]))
    irradiating.append(float(r[4]))

  dec_index = []
  dec_temperature = []
  dec_voltage = []
  dec_current = []
  dec_irradiating = []
  for r in dec_record:
    dec_temperature.append(float(r[1]))
    dec_voltage.append(float(r[2]))
    dec_current.append(float(r[3]))
    dec_irradiating.append(float(r[4]))

  fig, ax = plt.subplots(2, 2, layout="constrained")
  plots = plt.subplot(221)
  plt.plot(index, current)
  plt.plot(index, dec_current)
  plt.grid(True)
  plt.title("current")

  plots = plt.subplot(222)
  plt.plot(index, temperature)
  plt.plot(index, dec_temperature)
  plt.grid(True)
  plt.title("temperature")

  plots = plt.subplot(223)
  plt.plot(index, voltage)
  plt.plot(index, dec_voltage)
  plt.grid(True)
  plt.title("voltage")

  plots = plt.subplot(224)
  plt.plot(index, irradiating)
  plt.plot(index, dec_irradiating)
  plt.grid(True)
  plt.title("irradiating")

  if show:
    plt.show()
  if save:
    plt.savefig(filename)

  plt.clf()
  plt.cla()
  plt.close()

def draw_all(records, decoded_records, path, show=True, save=False):
  i = 0
  num_len = len(str(len(records)))
  while i < len(records):
    str_i = ''
    j = num_len - len(str(i+1))
    while j > 0:
      str_i = str_i + '0'
      j = j - 1
    str_i = str_i + str(i+1)
    filename = path + '/graphs/fig' + str_i+ '.png'
    draw_record(records[i], decoded_records[i], show, filename, save)
    i = i + 1

def save_to_csv(records, filename, path):
  f = open(path + '/' + filename, 'w')
  for val in records:
    for row in val:
      r = str(float(row[0][0]))+';'+str(float(row[1][0]))+';'+str(float(row[2][0]))+';'+str(float(row[3][0]))+';'+str(float(row[4][0]))+'\n'
      f.write(r)
  f.close()

def save_info(vals, dec_vals, path):
  os.mkdir(path)
  os.mkdir(path + '/graphs')
  save_to_csv(vals, path + '.csv', path)
  save_to_csv(dec_vals, path + '_decoded.csv', path)
  draw_all(vals, dec_vals, path, show=False, save=True, )

def split_data(data_arr):
  x_input = []
  x_output = []
  
  for r in data_arr:
    input_r = []
    output_r = []
    for line in r:
      input_r.append([line[0], line[1], line[4]])
      output_r.append([line[2], line[3]*10])
    x_input.append(input_r)
    x_output.append(output_r)
    
  x_input = np.asarray(x_input)
  x_output = np.asarray(x_output)
  x_input = x_input.astype('float32')
  x_output = x_output.astype('float32')
  
  return x_input, x_output

def concat_data(input_1, input_2):
  out_data = []
  i = 0
  while i < len(input_1):
    out_line = []
    j = 0
    while j < len(input_1[i]):
      out_line.append([input_1[i][j][0], input_1[i][j][1], input_2[i][j][0], input_2[i][j][1]/10, input_1[i][j][2]])
      j = j+1
    out_data.append(out_line)
    i = i+1
    
  out_data = np.asarray(out_data)
  out_data = out_data.astype('float32')
  
  return out_data


latent_dim = 384
EPOCHS_NUM = 1

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(600, activation='sigmoid'),
    ])
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='linear'),
    ])
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid'),
    ])
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(192, activation='sigmoid'),
    ])
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid'),
    ])
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='linear'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(400, activation='linear'),
      layers.Reshape((200,2,1))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

### Begin

x_test, x_train = get_temperature()

x_input, x_output = split_data(x_train)
y_input, y_output = split_data(x_test)

autoencoder = Autoencoder(latent_dim)

tensorboard = tf.keras.callbacks.TensorBoard(
  histogram_freq=1,
  write_graph=True,
  update_freq=1
  )

checkpoint_filepath = './weights/weights_solar.h5'
weight_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_filepath,
  monitor='loss',
  save_weights_only=True,
  save_best_only=True,
  mode='min',
  verbose=1
  )

callback_list = [weight_checkpoint_callback, tensorboard]

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(x_input, x_output,
                          epochs=EPOCHS_NUM,
                          shuffle=True,
                          validation_data=(y_input, y_output),
                          callbacks=callback_list).history
