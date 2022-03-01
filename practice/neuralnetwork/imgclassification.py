import zipfile, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split = 0.4
)

base_dir = 'rps-cv-images'

train_generator = train_datagen.flow_from_directory(
    base_dir, 
    target_size=(150, 150), 
    class_mode='categorical',
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir, 
    target_size=(150, 150), 
    class_mode='categorical',
    subset = 'validation'
)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.9):
      print("\nAkurasi Model sudah > 90%")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop' , metrics=['accuracy'])

model.fit(
      train_generator,
      steps_per_epoch=4, 
      epochs=100, 
      validation_data=validation_generator, 
      validation_steps=4,  
      verbose=2,
      callbacks=[callbacks]
)


image_path = 'testcase/image1.png'

img = image.load_img(image_path, target_size=(150, 150))                            
imgplot = plt.imshow(img)                                                     
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

if classes[0][0]==1:
    print('Kertas')                                 
elif classes[0][1]==1:
    print('Batu')                                   
elif classes[0][2]==1:
    print('Gunting')                                
else:
    print('Tidak Diketahui')                                                      
