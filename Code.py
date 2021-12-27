# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:55:09 2020

@author: saura
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import random

path = "/home/ss7876/Augmented_all/"
train_path = path+"/train"
test_path= path +"/val"

path = "/home/ss7876/"
artists = pd.read_csv(path+'/artists.csv')
artists.shape
artists_top = artists.sort_values(by=['paintings'],ascending=False).reset_index()
#artists_top = artists[artists['paintings']>100]
artists_top = artists_top[['name','paintings']]
artists_top['class_weight'] = artists_top.paintings.sum()/(artists_top.shape[0]*artists_top.paintings)
artists_top

artist = {}
artist_name=[]
for dirname, i, filenames in os.walk(train_path):
  artist_name = i
  print(i)
  break

artist_name = artist_name[0:50]
print(len(artist_name))
class_weights={}
for i in range(artists.shape[0]):
  name = artists_top.iloc[i,0].replace(" ","_")
  if name in artist_name:
    print("i:%d,name:%s"%(i,name))
    class_weights[name]=artists_top.iloc[i,2]
class_weights['Albrecht'] = artists_top.iloc[4,2]
print(class_weights)

class_weights_list = []
for name in artist_name:
  class_weights_list.append(class_weights[name])
class_weights_dict = {}
for i in range(50):
  class_weights_dict[i] = class_weights_list[i]
print(class_weights_dict)

n = 5
fig,axes = plt.subplots(1,n,figsize=(20,10))
for i in range(n):
  random_artist = artist_name[i]
  random_image = random.choice(os.listdir(os.path.join(train_path,random_artist)))
  random_image_file = os.path.join(train_path,random_artist,random_image)
  image = plt.imread(random_image_file)
  axes[i].imshow(image)
  axes[i].set_title("Artist:"+random_artist)
  axes[i].axis('off')
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 16 
train_input_shape = (224,224,3)
n_class = len(artist_name)
train_datagen = ImageDataGenerator(rescale = 1./255.,)
train_generator = train_datagen.flow_from_directory(directory=train_path,
                                                    target_size = train_input_shape[0:2],
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    subset = "training",
                                                    shuffle=True,
                                                    classes=artist_name)
valid_generator = train_datagen.flow_from_directory(directory=test_path,class_mode = 'categorical',
                                                    target_size = train_input_shape[0:2],
                                                    batch_size = batch_size,
                                                    subset = "training",
                                                    shuffle = True,
                                                    classes = artist_name)
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = train_generator.n//valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

#from keras.applications.xception import Xception
import tensorflow as tf
from tensorflow.python.keras.layers. normalization import BatchNormalization
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *


based_model = ResNet101(weights='imagenet', include_top=False, input_shape=train_input_shape)
#based_model = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=train_input_shape)

for layer in based_model.layers:
    layer.trainable = True
# Add layers at the end
X = based_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
#X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

X = Dense(128, kernel_initializer='he_uniform')(X)
#X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(n_class, activation='softmax')(X)

model = Model(inputs=based_model.input, outputs=output)

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
n_epoch = 15

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, 
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, 
                              verbose=1, mode='auto')


history1 = model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,
                               validation_data = valid_generator,validation_steps = STEP_SIZE_VALID,
                               epochs=n_epoch,
                               shuffle=True,
                               verbose = 1,
                               use_multiprocessing=True,
                               callbacks=[reduce_lr],
                               workers=16,
                               class_weight=class_weights_dict
                               )


for layer in model.layers:
    layer.trainable = False

for layer in model.layers[:101]:
    layer.trainable = True

optimizer = Adam(learning_rate=0.0001)
#optimizer = Adam(learning_rate=lr_schedule(0))
#lr_scheduler = LearningRateScheduler(lr_schedule)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])

n_epoch = 20
history2 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr, early_stop],
                              use_multiprocessing=True,
                              workers=16,
                              class_weight=class_weights_dict
                             )


score = model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on TEST data =", score[1])

score = model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on train data =", score[1])

history = {}
history['loss'] = history1.history['loss'] + history2.history['loss']
history['acc'] = history1.history['acc'] + history2.history['acc']
history['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']
history['val_acc'] = history1.history['val_acc'] + history2.history['val_acc']
history['lr'] = history1.history['lr'] + history2.history['lr']

# Plot the training graph
def plot_training(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')
    
    plt.show()
    
plot_training(history)