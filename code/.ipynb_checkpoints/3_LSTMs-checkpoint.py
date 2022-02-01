#!/usr/bin/env python
# coding: utf-8

# # Detection of composite transposable elements 
# ## Notebook 3: Testing TF Models
# ## Description:
# Transposable elements are sequences in genomes that can change their position in the genome and that is also the reason why they are colloquially called “jumping genes”. They are able to affect the composition and size of genetic replicons. Our research interest in this project are composite transposable elements, which are flanked by two inverted repeats and transposable elements. Composite transposable elements are moving as one unit within a genome and are copying and inserting genes enclosed by itself. The following traits of composite transposable elements are making the detection challenging: 
# 
# 1. Sometimes terminal information such as repeats or transposable elements are missing, which would define the boundaries of a composite transposable element.
# 2. Composite transposable elements are diverse in their genetic composition and size. 
# 
# Composite transposable elements are usually associated with essential and indispensable genes, which are having a high gene frequency across genomes. We hypothesize that the genetic frequency landscape of a genetic element will follow a particular pattern, which can be used as a marker for putative regions of composite transposable elements. Thus, we are representing here an approach to detect regions of putative composite transposable elements using gene frequencies, protein family clusters and positions of composite transposable elements as input for a supervised LSTM-based neural network model. 

# ### Project Repo 
# * add own github link

# ## Participants:
# Dustin Martin Hanke: dhanke@ifam.uni-kiel.de
# 
# Wang Yiging: ywang@ifam.uni-kiel.de 

# ### Course and Semester
# Machine Learning with TensorFlow - Wintersemester 2021/2022

# ### License
# If you are releasing the software under some certain license, you can mention it and also include the `LICENSE.md` file in the folder
# 
# ---

# ### Imports:

# In[1]:


# Imports:
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf
import tensorflow.keras.utils as ku 
import matplotlib.pyplot as plt
import numpy as np 
import random
import os

# Optimizers and learning rate
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import datetime

#Standard Scaler
from sklearn.preprocessing import StandardScaler


# ### Load input data:

# In[2]:


#Arrays have been stored as 1D-arrays
two_d_4 = np.loadtxt('../arrays/two_d_4.csv', delimiter=',').reshape(29360, 25, 2)
labels4 = np.loadtxt('../arrays/labels4.csv', delimiter=',').reshape(29360, 1)
print(two_d_4.shape, labels4.shape)


# ### Split data:

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(two_d_4, labels4, test_size=0.2, random_state=42, stratify=labels4)
print("The shapes of input and labels:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("Distribution:")
print("Positive samples: {}%".format((np.sum(y_train)/len(X_train))*100))
print("Negative samples: {}%".format((np.sum(y_test)/len(X_test))*100))


# # Baseline Models:

# In[4]:


def Baseline1():
    model = Sequential()
    #model.add(BiDirectional(LSTM(32, return_sequences=True), input_shape=(25, 2)))
    model.add(LSTM(32, return_sequences=True, input_shape=(25, 2)))
    model.add(Dense(units=1, activation="relu"))
    return model

def Baseline2():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(25, 2)))
    model.add(LSTM(32))
    model.add(Dense(units=1, activation="sigmoid"))
    return model

def Baseline3():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(25, 2)))
    model.add(LSTM(32))
    model.add(Dense(units=1, activation="sigmoid"))
    return model

def Baseline4():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(25, 2)))
    model.add(LSTM(32))
    model.add(Dense(units=16, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model

baseline1 = Baseline1()
baseline2 = Baseline2()
baseline3 = Baseline3()
baseline4 = Baseline4()


# ### Model compilation:
def compile_train(model, batch_size, num_epochs):
    num_train_steps = (len(y_train) // batch_size) * num_epochs
    print(num_train_steps)
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=6e-5, end_learning_rate=0., decay_steps=num_train_steps)
    updatelr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    opt = Adam(learning_rate=lr_scheduler)
    
    learning_rates = []
    for step in range(1, num_train_steps+1):
        decay_steps = num_train_steps
        step = min(step, decay_steps)
        learning_rates.append(((6e-7 - 0. ) * (1 - step / num_train_steps) ** (1.0)) + 0.)
    
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-14)
    #callback = tf.keras.callbacks.schedules.Pol(lambda epoch: 1e-3 * 10 ** (epoch / 30))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["learning_rate"])
    model.summary()
    # tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # save weights callback
    checkpoint_path = "training_2/biLSTM_composite_TE.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    
    print(opt.lr)
    
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, shuffle=True, callbacks=[cp_callback]) #reduce_lr for Plateau approach
    
    return history#, learning_rates 
# In[71]:


def compile_train(model, batch_size, num_epochs):
    num_train_steps = (len(y_train) // batch_size) * num_epochs
    print(num_train_steps)
    #lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=6e-5, end_learning_rate=0., decay_steps=num_train_steps)
    #updatelr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    opt = Adam()

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()
    # tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # save weights callback
    checkpoint_path = "training_2/biLSTM_composite_TE.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch/30))
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, shuffle=True, callbacks=[callback]) #reduce_lr for Plateau approach
    
    return history#, learning_rates 


# # Plot functions:

# In[31]:


def accuracy_plt(history, x):
    plt.figure(x)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Testing accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig("graphs/accuracy_{}.png".format(datetime.datetime.now()))
    #plt.show()
    return
# plot history
def loss_plt(history, x):
    plt.figure(x)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Testing loss')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig("graphs/loss_{}.png".format(datetime.datetime.now()))
    #plt.show()
    return

def rate_plt(history, x):
    plt.figure(x)
    plt.plot(history.history['lr'], label='Learning rate', color='#000')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig("graphs/learning_rate_{}.png".format(datetime.datetime.now()))
    #plt.show()
    return

def rate_loss(history, lrs, y_train_len, num_epochs, x):
    plt.figure(x, figsize=(18,8))
    new_lrs = [lrs[x] for x in range(0, len(lrs), int(y_train_len/num_epochs))]
    #xscale("log")
    plt.semilogx(new_lrs, history.history['loss'], lw=3, color='#000')
    #plt.xlim([6e-6, 0])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    #plt.ylim([0, 1])
    #plt.legend()
    plt.savefig("graphs/learning_rate_{}.png".format(datetime.datetime.now()))
    #plt.show()
    return


# # Execution:

# In[ ]:


#history, lrs = compile_train(baseline1, 32, 100)
#accuracy_plt(history, 1)
#loss_plt(history, 2)
#rate_loss(history, lrs, len(y_train), 32, 11)


# In[ ]:


#history, lrs = compile_train(baseline2, 32, 100)
#accuracy_plt(history, 3)
#loss_plt(history, 4)
#rate_loss(history, lrs, len(y_train), 32, 12)


# In[65]:


history = compile_train(baseline3, 32, 100)
accuracy_plt(history, 5)
loss_plt(history, 6)

learning_rates = 1e-6 * (10 ** (np.arange(100)/30))
plt.figure(20, figsize=(18,8))
plt.semilogx(learning_rates, history.history['loss'], lw=3, color='#000')
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.savefig("graphs/learning_rate_{}.png".format(datetime.datetime.now()))
#rate_loss(history, lrs, len(y_train), 32, 10)

