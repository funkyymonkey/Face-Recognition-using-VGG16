#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# STEP 1: PREPARE THE DATA 


# In[2]:


#IMAGE AUGUMENTATION

from keras.preprocessing.image import ImageDataGenerator

train_path = 'C:/Users/Ashutosh/Desktop/New folder/Celebrity-faces/data/train'
test_path = 'C:/Users/Ashutosh/Desktop/New folder/Celebrity-faces/data/val'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   rotation_range=45,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   zoom_range = 0.2,
                                   fill_mode='nearest',
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[3]:


#STEP 2: MODIFY THE PRE-TRAINED MODEL


# In[4]:


# import the pre trained architecture
from keras.applications.vgg16 import VGG16


# In[5]:


# load the model withut its top layers

model = VGG16(weights = 'imagenet', 
              include_top = False,
              input_shape = (224,224, 3))


# In[6]:


# set all the layers untrainable
# we dont want to change the weights

for layer in model.layers:
    layer.trainable = False


# In[7]:


# print all the layer name 

for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[8]:


#STEP 3: PREPARE THE TOP LAYERS OF FINAL MODEL


# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D


# In[10]:


model_top=model.output
model_top = GlobalAveragePooling2D()(model_top)
model_top = Dense(1024,activation='relu')(model_top)
model_top = Dense(1024,activation='relu')(model_top)
model_top = Dense(512,activation='relu')(model_top)
model_top = Dense(5,activation='softmax')(model_top)


# In[11]:


# STEP 4 : MERGE BOTH THE PARTS FOE FINAL MODEL


# In[12]:


from keras.models import Model
model_final=Model(inputs= model.input, outputs = model_top)


# In[13]:


model_final.summary()


# In[14]:


#STEP 5: MODEL FITTING


# In[16]:


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("face_recog.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]


# We use a very small learning rate 
model_final.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 93
nb_validation_samples = 25

# We only train 5 EPOCHS 
epochs = 8
batch_size = 15

history = model_final.fit_generator(
    training_set,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = test_set,
    validation_steps = nb_validation_samples // batch_size)


# In[ ]:


# STEP 6: LOAD THE MODEL 


# In[17]:


from keras.models import load_model

classifier = load_model('face_recog.h5')


# In[18]:


# STEP 7: VIEW THE RESULT


# In[ ]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

celeb_dict = {"[0]": 'ben_afflek', 
              "[1]": 'elton_john',
              "[2]": 'jerry_seinfeld',
              "[3]": 'madonna',
              "[4]": 'mindy_kaling'   }

celeb_dict_n = {"n0": 'ben_afflek', 
                "n1": 'elton_john',
                "n2": 'jerry_seinfeld',
                "n3": 'madonna',
                "n4": 'mindy_kaling'   }

def draw_test(name, pred, im):
    celeb = celeb_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, celeb, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    #print("Class - " + celeb_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("Celebrity-faces/data/val/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:




