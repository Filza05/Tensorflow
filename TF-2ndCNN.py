import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras

#split the dataset manually into 80% training, 10% testing and 10% validation data
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split = ['train[:80%]','train[80%:90%]','train[90%:]'],
    with_info = True,
    as_supervised=True,
)
get_label_name = metadata.features['label'].int2str

#Display 2 images from the dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    
IMG_SIZE = 160
def format_example(image, label):
    image = tf.cast(image,tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image, label in raw_train.take(2):
    print("ORIGINAL SHAPE: ", image.shape)
    
for image, label in train.take(2):
    print("NEW SHAPE: ", image.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras,applications.MobileNetV2(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights = 'imagenet'
)

for image, _ in train_batches.take(1):
    pass

    feature_batch = base_model(image)
    print(feature_batch.shape)
    
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.summary()

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches,steps=validation_steps)

#Now we can train it on our images
history = model.fit(
    train_batches,
    epochs=initial_epochs,
    validation_data=validation_batches
)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")
#h5 is standard to saving models in keras, not tf, in keras only
new_model = tf.keras.models.load_model("dogs_vs_cats.h5")