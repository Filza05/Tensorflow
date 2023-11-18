import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images / 255.0 
test_images = test_images / 255.0

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

im_idx = 3
plt.imshow(train_images[im_idx],cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[im_idx][0]])
plt.show()

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10))  #coz we have 10 classes

model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

history = model.fit(train_images,train_labels,epochs=10,
                    validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print(test_acc*100)

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#creates a datagen object that transforms images
dataGen = ImageDataGenerator (
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#pick img to transform
testImage = train_images[10]
img = image.img_to_array(testImage) #convert image to numpy array
img = img.reshape((1,) + img.shape) #reshape image

i = 0

for batch in dataGen.flow(img,save_prefix='test',save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:
        break
    
plt.show()