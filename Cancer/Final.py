#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import time

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3


# In[2]:


#1. Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# In[3]:


base_skin_dir = os.path.join('C:/Users/suhas/Documents/Datacamp projects', 'Cancer')

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[4]:




skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


# In[5]:


# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()


# In[6]:


skin_df.isnull().sum()


# In[7]:


skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)


# In[8]:


skin_df.isnull().sum()


# In[9]:


print(skin_df.dtypes)


# In[10]:


fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)


# In[11]:


skin_df['dx_type'].value_counts().plot(kind='bar')


# In[12]:


skin_df['localization'].value_counts().plot(kind='bar')


# In[13]:


skin_df['age'].hist(bins=40)


# In[14]:



skin_df['sex'].value_counts().plot(kind='bar')


# In[15]:


sns.scatterplot('age','cell_type_idx',data=skin_df)


# In[16]:


sns.factorplot('sex','cell_type_idx',data=skin_df)


# In[ ]:





# In[17]:


skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# In[18]:


skin_df.head()


# In[19]:


n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)


# In[20]:


# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()


# In[21]:


features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']


# In[22]:


input_shape = (75, 100, 3)
num_classes = 7


# In[23]:


################################
#Creating 10% training data
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.90,random_state=108)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
# Perform one-hot encoding on the labels
y_train_10 = to_categorical(y_train_o, num_classes = 7)
y_test_10 = to_categorical(y_test_o, num_classes = 7)

x_train_10 = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test_10 = x_test.reshape(x_test.shape[0], *(75, 100, 3))


# In[24]:


################################
#Creating 20% training data

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.80,random_state=108)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
# Perform one-hot encoding on the labels
y_train_20 = to_categorical(y_train_o, num_classes = 7)
y_test_20 = to_categorical(y_test_o, num_classes = 7)

x_train_20 = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test_20 = x_test.reshape(x_test.shape[0], *(75, 100, 3))


# In[25]:


################################
#Creating 30% training data

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.70,random_state=108)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
# Perform one-hot encoding on the labels
y_train_30 = to_categorical(y_train_o, num_classes = 7)
y_test_30 = to_categorical(y_test_o, num_classes = 7)

x_train_30 = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test_30 = x_test.reshape(x_test.shape[0], *(75, 100, 3))


# In[26]:


################################
#Creating 40% training data

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.60,random_state=108)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
# Perform one-hot encoding on the labels
y_train_40 = to_categorical(y_train_o, num_classes = 7)
y_test_40 = to_categorical(y_test_o, num_classes = 7)

x_train_40 = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test_40 = x_test.reshape(x_test.shape[0], *(75, 100, 3))


# In[27]:


################################
#Creating 50% training data

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.50,random_state=108)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
# Perform one-hot encoding on the labels
y_train_50 = to_categorical(y_train_o, num_classes = 7)
y_test_50 = to_categorical(y_test_o, num_classes = 7)

x_train_50 = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test_50 = x_test.reshape(x_test.shape[0], *(75, 100, 3))


# In[28]:


# Training the model for 10% training size
m = []
acc1 = []
start = time.time()

K.clear_session()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 32
epochs = 12

history = model.fit(x_train_10, y_train_10,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_10, y_test_10))

score = model.evaluate(x_test_10, y_test_10, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

m.append(end-start)
acc1.append(score[1])


# In[29]:


# Training the model for 20% training size

start = time.time()

K.clear_session()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 32
epochs = 12

history = model.fit(x_train_20, y_train_20,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_20, y_test_20))

score = model.evaluate(x_test_20, y_test_20, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

m.append(end-start)
acc1.append(score[1])


# In[30]:


# Training the model for 30% training size

start = time.time()


K.clear_session()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 32
epochs = 12

history = model.fit(x_train_30, y_train_30,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_30, y_test_30))

score = model.evaluate(x_test_30, y_test_30, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

m.append(end-start)

acc1.append(score[1])


# In[31]:


# Training the model for 40% training size
start = time.time()

K.clear_session()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 32
epochs = 12

history = model.fit(x_train_40, y_train_40,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_40, y_test_40))

score = model.evaluate(x_test_40, y_test_40, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

m.append(end-start)

acc1.append(score[1])


# In[32]:


# Training the model for 50% training size

start = time.time()

K.clear_session()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 32
epochs = 12

history = model.fit(x_train_50, y_train_50,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_50, y_test_50))

score = model.evaluate(x_test_50, y_test_50, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

m.append(end-start)

acc1.append(score[1])


# In[34]:


# Training the VGG16 model for 10% training size
n = []
acc2 = []
start = time.time()

K.clear_session()

tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(x_train_10)

annealer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()

tmodel.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history = tmodel.fit(x_train_10,y_train_10,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_10,y_test_10),
                    verbose=1)

score = tmodel.evaluate(x_test_10, y_test_10, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc2.append(score[1])
n.append(end-start)


# In[35]:


# Training the VGG16 model for 20% training size

start = time.time()

K.clear_session()

tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(x_train_20)

annealer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()

tmodel.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history = tmodel.fit(x_train_20,y_train_20,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_20,y_test_20),
                    verbose=1)

plot_model_history(history)

score = tmodel.evaluate(x_test_20, y_test_20, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc2.append(score[1])

n.append(end-start)


# In[36]:


# Training the VGG16 model for 30% training size

start = time.time()

K.clear_session()

tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(x_train_30)

annealer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()

tmodel.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history = tmodel.fit(x_train_30,y_train_30,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_30,y_test_30),
                    verbose=1)

plot_model_history(history)
score = tmodel.evaluate(x_test_30, y_test_30, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc2.append(score[1])

n.append(end-start)


# In[37]:


# Training the VGG16 model for 40% training size

start = time.time()

K.clear_session()

tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(x_train_40)

annealer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()

tmodel.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history = tmodel.fit(x_train_40,y_train_40,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_40,y_test_40),
                    verbose=1)

plot_model_history(history)

score = tmodel.evaluate(x_test_40, y_test_40, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc2.append(score[1])

n.append(end-start)


# In[39]:


# Training the VGG16 model for 50% training size

start = time.time()

K.clear_session()

tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(x_train_50)

annealer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()

tmodel.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history = tmodel.fit(x_train_50,y_train_50,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_50,y_test_50),
                    verbose=1)

plot_model_history(history)

score = tmodel.evaluate(x_test_50, y_test_50, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


end = time.time()

acc2.append(score[1])

n.append(end-start)


# In[40]:


# Training the InceptionV3 model for 10% training size

o = []
acc3 = []
start = time.time()

K.clear_session()

model1 = InceptionV3(include_top=False, 
                     weights='imagenet', 
                     input_tensor=None, 
                     input_shape=input_shape, 
                     pooling=None, 
                     classes=num_classes)

augs1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs1.fit(x_train_10)

annealer1 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel1 = Sequential()
tmodel1.add(model1)
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.50))
tmodel1.add(Flatten())
tmodel1.add(Dense(512,activation='relu'))
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.25))
tmodel1.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel1.summary()

tmodel1.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history1 = tmodel1.fit(x_train_10,y_train_10,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_10,y_test_10),
                    verbose=1)

plot_model_history(history1)

score = tmodel1.evaluate(x_test_10, y_test_10, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc3.append(score[1])
o.append(end-start)


# In[41]:


# Training the InceptionV3 model for 20% training size

start = time.time()

K.clear_session()

model1 = InceptionV3(include_top=False, 
                     weights='imagenet', 
                     input_tensor=None, 
                     input_shape=input_shape, 
                     pooling=None, 
                     classes=num_classes)

augs1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs1.fit(x_train_20)

annealer1 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel1 = Sequential()
tmodel1.add(model1)
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.50))
tmodel1.add(Flatten())
tmodel1.add(Dense(512,activation='relu'))
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.25))
tmodel1.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel1.summary()

tmodel1.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history1 = tmodel1.fit(x_train_20,y_train_20,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_20,y_test_20),
                    verbose=1)

plot_model_history(history1)

score = tmodel1.evaluate(x_test_20, y_test_20, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

acc3.append(score[1])
o.append(end-start)


# In[42]:


# Training the InceptionV3 model for 30% training size

start = time.time()

K.clear_session()

model1 = InceptionV3(include_top=False, 
                     weights='imagenet', 
                     input_tensor=None, 
                     input_shape=input_shape, 
                     pooling=None, 
                     classes=num_classes)

augs1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs1.fit(x_train_30)

annealer1 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel1 = Sequential()
tmodel1.add(model1)
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.50))
tmodel1.add(Flatten())
tmodel1.add(Dense(512,activation='relu'))
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.25))
tmodel1.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel1.summary()

tmodel1.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history1 = tmodel1.fit(x_train_30,y_train_30,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_30,y_test_30),
                    verbose=1)

plot_model_history(history1)

score = tmodel1.evaluate(x_test_30, y_test_30, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc3.append(score[1])

o.append(end-start)


# In[43]:


# Training the InceptionV3 model for 40% training size

start = time.time()

K.clear_session()

model1 = InceptionV3(include_top=False, 
                     weights='imagenet', 
                     input_tensor=None, 
                     input_shape=input_shape, 
                     pooling=None, 
                     classes=num_classes)

augs1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs1.fit(x_train_40)

annealer1 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel1 = Sequential()
tmodel1.add(model1)
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.50))
tmodel1.add(Flatten())
tmodel1.add(Dense(512,activation='relu'))
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.25))
tmodel1.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel1.summary()

tmodel1.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history1 = tmodel1.fit(x_train_40,y_train_40,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_40,y_test_40),
                    verbose=1)

plot_model_history(history1)

score = tmodel1.evaluate(x_test_40, y_test_40, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc3.append(score[1])

o.append(end-start)


# In[44]:


# Training the InceptionV3 model for 50% training size

start = time.time()

K.clear_session()

model1 = InceptionV3(include_top=False, 
                     weights='imagenet', 
                     input_tensor=None, 
                     input_shape=input_shape, 
                     pooling=None, 
                     classes=num_classes)

augs1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs1.fit(x_train_50)

annealer1 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)


tmodel1 = Sequential()
tmodel1.add(model1)
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.50))
tmodel1.add(Flatten())
tmodel1.add(Dense(512,activation='relu'))
tmodel1.add(BatchNormalization())
tmodel1.add(Dropout(0.25))
tmodel1.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel1.summary()

tmodel1.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history1 = tmodel1.fit(x_train_50,y_train_50,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_50,y_test_50),
                    verbose=1)

plot_model_history(history1)

score = tmodel1.evaluate(x_test_50, y_test_50, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

acc3.append(score[1])

o.append(end-start)


# In[46]:


from keras.applications.xception import Xception


# In[47]:


# Training the Xception model for 10% training size
p = []
acc4 = []

start = time.time()

K.clear_session()

model2 = Xception(include_top=False, 
                  weights='imagenet', 
                  input_tensor=None, 
                  input_shape=input_shape, 
                  pooling=None, 
                  classes=num_classes)
augs2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs2.fit(x_train_10)

annealer2 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)
tmodel2 = Sequential()
tmodel2.add(model2)
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.50))
tmodel2.add(Flatten())
tmodel2.add(Dense(512,activation='relu'))
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.25))
tmodel2.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel2.summary()

tmodel2.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history2 = tmodel2.fit(x_train_10,y_train_10,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_10,y_test_10),
                    verbose=1)

plot_model_history(history2)
score = tmodel2.evaluate(x_test_10, y_test_10, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc4.append(score[1])
p.append(end-start)


# In[48]:


# Training the Xception model for 20% training size

start = time.time()

K.clear_session()

model2 = Xception(include_top=False, 
                  weights='imagenet', 
                  input_tensor=None, 
                  input_shape=input_shape, 
                  pooling=None, 
                  classes=num_classes)
augs2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs2.fit(x_train_20)

annealer2 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)
tmodel2 = Sequential()
tmodel2.add(model2)
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.50))
tmodel2.add(Flatten())
tmodel2.add(Dense(512,activation='relu'))
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.25))
tmodel2.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel2.summary()

tmodel2.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history2 = tmodel2.fit(x_train_20,y_train_20,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_20,y_test_20),
                    verbose=1)

plot_model_history(history2)

end = time.time()
score = tmodel2.evaluate(x_test_20, y_test_20, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

acc4.append(score[1])
p.append(end-start)


# In[49]:


# Training the Xception model for 30% training size

start = time.time()

K.clear_session()

model2 = Xception(include_top=False, 
                  weights='imagenet', 
                  input_tensor=None, 
                  input_shape=input_shape, 
                  pooling=None, 
                  classes=num_classes)
augs2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs2.fit(x_train_30)

annealer2 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)
tmodel2 = Sequential()
tmodel2.add(model2)
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.50))
tmodel2.add(Flatten())
tmodel2.add(Dense(512,activation='relu'))
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.25))
tmodel2.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel2.summary()

tmodel2.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history2 = tmodel2.fit(x_train_30,y_train_30,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_30,y_test_30),
                    verbose=1)

plot_model_history(history2)

score = tmodel2.evaluate(x_test_30, y_test_30, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

acc4.append(score[1])

p.append(end-start)


# In[50]:


# Training the Xception model for 40% training size

start = time.time()

K.clear_session()

model2 = Xception(include_top=False, 
                  weights='imagenet', 
                  input_tensor=None, 
                  input_shape=input_shape, 
                  pooling=None, 
                  classes=num_classes)
augs2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs2.fit(x_train_10)

annealer2 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)
tmodel2 = Sequential()
tmodel2.add(model2)
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.50))
tmodel2.add(Flatten())
tmodel2.add(Dense(512,activation='relu'))
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.25))
tmodel2.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel2.summary()

tmodel2.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history2 = tmodel2.fit(x_train_40,y_train_40,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_40,y_test_40),
                    verbose=1)

plot_model_history(history2)

score = tmodel2.evaluate(x_test_40, y_test_40, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()
acc4.append(score[1])

p.append(end-start)


# In[51]:


# Training the Xception model for 50% training size

start = time.time()

K.clear_session()

model2 = Xception(include_top=False, 
                  weights='imagenet', 
                  input_tensor=None, 
                  input_shape=input_shape, 
                  pooling=None, 
                  classes=num_classes)
augs2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs2.fit(x_train_50)

annealer2 = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)
tmodel2 = Sequential()
tmodel2.add(model2)
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.50))
tmodel2.add(Flatten())
tmodel2.add(Dense(512,activation='relu'))
tmodel2.add(BatchNormalization())
tmodel2.add(Dropout(0.25))
tmodel2.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel2.summary()

tmodel2.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])

batch_size = 128
epochs = 30

history2 = tmodel2.fit(x_train_50,y_train_50,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_50,y_test_50),
                    verbose=1)

plot_model_history(history2)

score = tmodel2.evaluate(x_test_50, y_test_50, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

end = time.time()

acc4.append(score[1])

p.append(end-start)


# In[52]:


# plotting


# In[65]:


plt.figure(figsize = (10,6))
plt.title("Time elapsed with each training size")
plt.plot([10,20,30,40,50], m,color = 'red', label = 'Defined')
plt.scatter([10,20,30,40,50], m,color = 'red')
plt.plot([10,20,30,40,50], n, label = 'VGG 16', color = 'orange')
plt.scatter([10,20,30,40,50], n, color = 'orange')
plt.plot([10,20,30,40,50], o, label = 'InceptionV3', color = 'green')
plt.scatter([10,20,30,40,50], o, color = 'green')
plt.plot([10,20,30,40,50], p, label = 'Xception', color ='red')
plt.scatter([10,20,30,40,50], p, color = 'red')
plt.xlabel('Training Size')
plt.ylabel('Time in seconds (s)')
plt.legend()
plt.show()


# In[67]:


plt.figure(figsize = (10,6))
plt.title("Validation Accuracy")
plt.plot([10,20,30,40,50], acc1, label = 'Defined', color ='red')
plt.scatter([10,20,30,40,50], acc1,color = 'red')
plt.plot([10,20,30,40,50], acc2, label = 'VGG 16', color = 'orange')
plt.scatter([10,20,30,40,50], acc2, color = 'orange')
plt.plot([10,20,30,40,50],acc3, label = 'InceptionV3', color = 'green')
plt.scatter([10,20,30,40,50], acc3, color = 'green')
plt.plot([10,20,30,40,50], acc4, label = 'Xception', color = 'red')
plt.scatter([10,20,30,40,50], acc4, color ='red')
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


print(m)
print(n)
print(o)
print(p)




