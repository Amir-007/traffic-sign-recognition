
from keras.models import Sequential
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense , Dropout

# initialize cnn
classifier = Sequential()

# convolution
classifier.add(Conv2D(filters = 32 ,kernel_size = (3 ,3) , input_shape= (64,64,3) , activation = 'relu'))

# Pooling
classifier.add(MaxPool2D(pool_size=(2,2) ))

# dropout
classifier.add(Dropout(rate=0.2))

# adding second convolution layer
classifier.add(Conv2D(filters = 32 ,kernel_size = (3 ,3) , activation = 'relu'))
classifier.add(MaxPool2D(pool_size=(2,2) ))
classifier.add(Dropout(rate=0.2))

# Flatten
classifier.add(Flatten())

# fully connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units=43,activation='softmax'))

# compiling the cnn
classifier.compile(optimizer='adam' , loss='categorical_crossentropy',metrics=['accuracy'])

# fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

classifier.summary()

history= classifier.fit_generator(training_set ,
               steps_per_epoch=39209//32,
               epochs=15,
               validation_data=test_set,
               validation_steps=12630//32)

# visualising the accuracy/loss on each epoch
import matplotlib.pyplot as plt
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

