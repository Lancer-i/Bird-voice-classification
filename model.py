
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

batch_size = 16

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,rotation_range=75)
test_datagen=ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'trainler',  
        target_size=(256,256),  
        batch_size=batch_size,
        classes = ['alakarga_yeni','boyunceviren_yeni','bulbul_yeni','çıvgın_yeni','çütre_yeni',
           'guguk_yeni','ibibik_yeni','karatavuk_yeni','kızılkuyruk_yeni','ormantoygarı_yeni'],
        class_mode='categorical')
test_generator= test_datagen.flow_from_directory(
        'valid', 
        target_size=(256,256),  
        batch_size=batch_size,
        classes = ['alakarga_yeni','boyunceviren_yeni','bulbul_yeni','çıvgın_yeni','çütre_yeni',
           'guguk_yeni','ibibik_yeni','karatavuk_yeni','kızılkuyruk_yeni','ormantoygarı_yeni'],
        class_mode='categorical')

import tensorflow as tf

model = tf.keras.models.Sequential([
 
   
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(512,512, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
  
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
import tensorflow
from tensorflow.keras.optimizers import RMSprop
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.001)
#RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,# adam
              metrics=['acc'])

total_sample=train_generator.n

n_epochs = 20
#steps_per_epoch=50,
history = model.fit_generator(
        train_generator, 
        steps_per_epoch=50,
        epochs=n_epochs,
        shuffle=True,
        validation_steps=int(total_sample/batch_size),
        validation_data=test_generator,
        verbose=1)

model.save('model_valid_4.h5')

