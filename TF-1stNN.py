import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
  
fashion_mnist = keras.datasets.fashion_mnist          
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

class_name = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([       #sequential is the simplest NN, passing from left to right
    keras.layers.Flatten(input_shape=(28,28)),  #input layer 1, flatten 28,28 shape into 784, one line kinda
    keras.layers.Dense(128,activation='relu'),  #hidden layer
    keras.layers.Dense(10,activation='softmax') #output layer
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images,train_labels,epochs=10)

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=1)
print("Test Accuracy Is:" , test_acc)

#predictions = model.predict(test_images)
#print(class_name[np.argmax(predictions[0])]) #argmax returns the index of highest value in the list

#plt.figure()
#plt.imshow(test_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

COLOR = 'White'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict (model, image, correct_label):
    prediction = model.predict(np.array([image]))
    predicted_class = class_name[np.argmax(prediction)]
    
    show_image(image,class_name[correct_label],predicted_class)
    
    
def show_image(img,label,guess):
    plt.figure()
    plt.imshow(img,cmap=plt.cm.binary)
    print('Expected: ' + label)
    print("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
def getNumber():
    while True:
        num = input('Pick a number:')
        if num.isdigit():
            num = int(num)
            if 0 <=num <= 1000:
                return int(num)
        else:
            print('Try Again...')
            
num = getNumber()
image = test_images[num]
label = test_labels[num]
predict(model,image,label)      
