import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np

dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
train_ds, test_ds = dataset['train'], dataset['test']
NUM_CLASSES = info.features['label'].num_classes
IMG_SIZE = 224

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights = 'imagenet')
base_model.trainable = False

model = models.Sequential([base_model, layers.GlobalAveragePooling2D(), layers.Dense(NUM_CLASSES, activation = 'softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=test_ds, epochs=5)

model.save("cat_classifier.h5")