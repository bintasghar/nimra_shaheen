import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk(r'The IQ-OTHNCCD lung cancer dataset - Copy (2)'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import cv2
import os
import imageio


import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import img_to_array, load_img

BATCH_SIZE = 96
IMAGE_SIZE = 256
EPOCHS=20
CHANNELS=3

 

for key in size_data:
  print(key, '->', size_data[key])

  image_classes = list(size_data.keys())
  number = list(size_data.values())

  print(image_classes)
  print(number)
  bengin_num = sum(size_data['Bengin cases'].values())
  malignant_num = sum(size_data['Malignant cases'].values())
  normal_num = sum(size_data['Normal cases'].values())

  print("Number of Benign Cases ->", bengin_num)
  print("Number of Malignant Cases ->", malignant_num)
  print("Number of Normal Cases ->", normal_num)

  num_list = [bengin_num, malignant_num, normal_num]
  print(num_list)
  all_image = malignant_num + bengin_num + normal_num
  print("Total Number of Images ->", all_image)
  c = ['blue', 'orange', 'green']
  # define the figure size
  plt.rcParams["figure.figsize"] = (10, 6)
  # make the barchart
  plt.bar(range(len(num_list)), num_list, tick_label=image_classes, color=c)
  # set the title and labels
  plt.suptitle('Number of cases on each class', y=0.95, fontsize=18)
  plt.xlabel('Cancer Classes', fontsize=16)
  plt.ylabel('Number of cases', fontsize=16)
  # set y-axis limit
  plt.ylim([0, 1400])
  plt.show()
  explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

  fig1, ax1 = plt.subplots()
  ax1.pie(num_list, explode=explode, labels=image_classes, autopct='%1.1f%%',
          shadow=True, startangle=90)

  ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.suptitle('Image Cases Percentages', y=1, fontsize=18)
  plt.legend(loc='lower right')

  plt.show()
  for image_batch, label_batch in directory.take(1):
      print(image_batch.shape)
      print(image_batch[1])
      print(label_batch.numpy())
  plt.figure(figsize=(15, 15))
  # iterate over the images in the shuffled dataset and take a sample consists of 16 images with their corresponding labels
  for image_batch, labels_batch in directory.take(1):
      for i in range(16):
          ax = plt.subplot(8, 8, i + 1)
          plt.imshow(image_batch[i].numpy().astype("uint8"))
          plt.title(categories[labels_batch[i]])
          plt.axis("off")
  plt.suptitle('Image Cases', y=1.05, fontsize=18)
  for i in categories:
      path = os.path.join(r'The IQ-OTHNCCD lung cancer dataset - Copy (2)', i)
      class_num = categories.index(i)
      for file in os.listdir(path):
          filepath = os.path.join(path, file)
          img = cv2.imread(filepath, 0)
          plt.imshow(img)
          plt.axis("off")
          plt.title(i)
          plt.show()
          break
  for i in categories:
      # take 3 samples from each class to be proccessed
      cnt, samples = 0, 3
      fig, ax = plt.subplots(samples, 4, figsize=(15, 15))
      fig.suptitle(i)

      path = os.path.join(r'The IQ-OTHNCCD lung cancer dataset - Copy (2)', i)
      class_num = categories.index(i)
      for curr_cnt, file in enumerate(os.listdir(path)):
          filepath = os.path.join(path, file)
          print(filepath)
          img = cv2.imread(filepath)
          cv2.imshow('windi', img)
          cv2.waitKey(0)

          # Convert to grayscals
          gray = cv2.imread(filepath, 0)

          # Resizing images with the target image size
          # img0 = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

          # Apply GaussianBlur to smooth the image (remove any spark noise that maybe affect the sharpening filters)
          blurred = cv2.GaussianBlur(gray, (5, 5), 0)

          # Apply Adaptive thresholding to get the high components (details or edges) in the image
          thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
          thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

          # Apply Bit Plane Slicing to select the image with highest details and neglect other planes
          lst = []
          for i in range(gray.shape[0]):
              for j in range(gray.shape[1]):
                  lst.append(np.binary_repr(gray[i][j], width=8))  # width = no. of bits

          eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(gray.shape[0],
                                                                                             gray.shape[1])

          # Apply Image Negative to reverse the image from white to black or vise versa
          arr = np.array(gray)
          invert = np.array(img)
          Max = np.max(invert)
          for i in range(invert.shape[0]):
              for j in range(invert.shape[1]):
                  invert[i, j] = Max - invert[i, j]

          # give a title for each image after applying the filter

          ax[cnt, 0].text(0.5, 0.5, 'Gray Scale Image', fontsize=19, bbox=dict(facecolor='red', alpha=0.5))
          ax[cnt, 0].imshow(img)
          ax[cnt, 1].text(0.5, 0.5, 'MSB Plane Image', fontsize=19, bbox=dict(facecolor='green', alpha=0.5))
          ax[cnt, 1].imshow(eight_bit_img)
          ax[cnt, 2].text(0.5, 0.5, 'Thresholding', fontsize=19, bbox=dict(facecolor='blue', alpha=0.5))
          ax[cnt, 2].imshow(thresh)
          ax[cnt, 3].text(0.5, 0.5, 'Image Negative', fontsize=19, bbox=dict(facecolor='yellow', alpha=0.5))
          ax[cnt, 3].imshow(invert)
          cnt += 1
          if cnt == samples:
              break
  plt.show()
  resize_and_rescale = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
      tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)])

  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.RandomRotation(0.2),
  ])

  def get_dataset_partitions_tf(ds, train_split=0.6, val_split=0.2, test_split=0.2, shuffle=True, shuffle_size=10000):
      assert (train_split + test_split + val_split) == 1
      ds_size = len(ds)

      # Shuffle the data
      if shuffle:
          ds = ds.shuffle(shuffle_size, seed=12)

      # split the data
      train_size = int(train_split * ds_size)
      val_size = int(val_split * ds_size)
      train_ds = ds.take(train_size)
      val_ds = ds.skip(train_size).take(val_size)
      test_ds = ds.skip(train_size).skip(val_size)

      # Autotune the dataset
      train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
      val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
      test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
      return train_ds, val_ds, test_ds

  # call the function to return train data, validation data, test data
  train_ds, val_ds, test_ds = get_dataset_partitions_tf(directory)
  input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
  num_classes = 3

  # Create the CNN model achitecture
  model = tf.keras.models.Sequential([
      # data_normalization
      data_augmentation,
      resize_and_rescale,

      # model layers
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D((2, 2)),

      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),

      tf.keras.layers.Dropout(0.25),

      # tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      # tf.keras.layers.MaxPooling2D((2, 2)),
      #
      # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      # tf.keras.layers.MaxPooling2D((2, 2)),

      # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      # tf.keras.layers.MaxPooling2D((2, 2)),
      #
      # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      # tf.keras.layers.MaxPooling2D((2, 2)),
      # tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(num_classes, activation='softmax'),
  ])
  model.build(input_shape=input_shape)

  class myCallback(tf.keras.callbacks.Callback):
      # Define the correct function signature for on_epoch_end
      def on_epoch_end(self, epoch, logs={}):
          if (logs.get('accuracy') >= 1.00):
              print("\nReached 100% accuracy so cancelling training!")

              # Stop training once the above condition is met
              self.model.stop_training = True


  # Instantiate the callback class
  callbacks = myCallback()
  optimizer = keras.optimizers.Adam(lr = 0.001)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  model.summary()
  history = model.fit(
      train_ds,
      batch_size=BATCH_SIZE,
      validation_data=val_ds,
      verbose=1,
      epochs=EPOCHS,
      callbacks=[callbacks]
  )
  plt.plot(history.history['accuracy'], label='Train')
  plt.plot(history.history['val_accuracy'], label='Validation')
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.show()

  # visualize the model loss for the training data and Validation data
  plt.plot(history.history['loss'], label='Train')
  plt.plot(history.history['val_loss'], label='Validation')
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend()
  plt.show()

  results = model.evaluate(test_ds)
  print("Model.metrics_names")
  print(results)

  # model.save("2Conv-NewParams.h5")

  image_path = r"C:\Users\shera\OneDrive\Desktop\The IQ-OTHNCCD lung cancer dataset - Copy (2)\Bengin cases\Bengin case (85).jpg"
  image = tf.keras.preprocessing.image.load_img(image_path)
  image_array = tf.keras.preprocessing.image.img_to_array(image)
  scaled_img = np.expand_dims(image_array, axis=0)
  plt.figure(figsize=(6, 6))
  plt.imshow(image)
  plt.axis("off")
  plt.title('output')

  # Use model to predict the sample image
  pred = model.predict(scaled_img)

  # show the output of predicted image
  output = categories[np.argmax(pred)]
  plt.show()
  print("Predicted case ->", output)
