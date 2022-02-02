#!/usr/bin/env python
# coding: utf-8

# # Detection of composite transposable elements 
# ## Notebook 3: Testing TF Models
# ## Description:
# Transposable elements are sequences in genomes that can change their position in the genome. Thus, they are also called “jumping genes”. They are able to affect the composition and size of genetic replicons. Our research interest in this project are composite transposable elements, which are flanked by two inverted repeats and transposable elements. Composite transposable elements are moving as one unit within a genome and are copying and inserting genes enclosed by itself. The following traits of composite transposable elements are making their detection challenging: 
# 
# 1. Sometimes terminal information such as repeats or transposable elements are missing, which would theoretically determine the boundaries of a composite transposable element.
# 2. Composite transposable elements are diverse in their genetic composition and size. 
# 
# Composite transposable elements are usually associated with essential and indispensable genes, which are having a high gene frequency across genomes, but also with genes of lower essentiality, which leads to significant drop in the gene frequency landscape. We hypothesize that the genetic frequency landscape of a replicon will follow a particular pattern, which can be used as a marker for putative regions of composite transposable elements. Thus, we are representing here an approach to detect regions of putative composite transposable elements using gene frequencies, protein family clusters and positions of composite transposable elements as input for a supervised LSTM-based neural network model. 

# ### Project Repo 
# https://github.com/DMH-dutte/Detection_of_composite_transposable_elements

# ## Participants:
# Dustin Martin Hanke: dhanke@ifam.uni-kiel.de
# 
# Wang Yiging: ywang@ifam.uni-kiel.de 

# ### Course and Semester
# Machine Learning with TensorFlow - Wintersemester 2021/2022

# ### Imports:

# In[99]:


from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import datetime
import random
import os


# ### Load input data:

# In[97]:


#Arrays have been stored as 1D-arrays and are reshaped directly into the correct format
two_d_4 = np.loadtxt('../arrays/two_d_4.csv', delimiter=',').reshape(29360, 25, 2)
labels4 = np.loadtxt('../arrays/labels4.csv', delimiter=',').reshape(29360, 1)
print("The shape of the input data is: {}".format(two_d_4.shape))
print("The shape of the labels is: {}".format(labels4.shape))


# ### Split data:

# In[98]:


X_train, X_test, y_train, y_test = train_test_split(two_d_4, labels4, test_size=0.2, random_state=42, stratify=labels4)
print("The shapes of the splitted input data and labels:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("Distribution:")
print("Positive samples: {}%".format((np.sum(y_train)/len(X_train))*100))
print("Negative samples: {}%".format((np.sum(y_test)/len(X_test))*100))


# # Plot functions:

# In[ ]:


def accuracy_plt(history, x):
    plt.figure(x)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Testing accuracy')
    plt.ylim([0, 1])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("graphs/accuracy_{}.png".format(datetime.datetime.now()))
    return

def loss_plt(history, x):
    plt.figure(x)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Testing loss')
    plt.ylim([0, 1])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("graphs/loss_{}.png".format(datetime.datetime.now()))
    return

def rate_loss_log(lrs, history, x):
    plt.figure(x, figsize=(18,8))
    plt.semilogx(lrs, history.history['loss'], lw=3, color='#000')
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig("graphs/learning_rate_{}.png".format(datetime.datetime.now()))
    return


# # Baseline Models:

# In[4]:


def Baseline1():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(25, 2)))
    model.add(Dense(units=1, activation="sigmoid"))
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


# ### Model compilation to find the optimal learning rates:

# In[108]:


lr_start = 1e-6 
epochs = 2


# In[109]:


def compile_train(model, batch_size, num_epochs, lr_start):
    num_train_steps = (len(y_train) // batch_size) * num_epochs
    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()
    
    # tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # save weights callback
    #checkpoint_path = "training_2/biLSTM_composite_TE.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    
    #Learning rate callback
    callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_start * 10 ** (epoch/30))
    
    #Fit model
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, shuffle=True, callbacks=[callback]) 
    
    return history


# # Execution to find the optimal learning rate for each model:

# In[110]:


learning_rates = lr_start * (10 ** (np.arange(epochs)/30))


# In[ ]:


history = compile_train(baseline1, 32, epochs, lr_start)
accuracy_plt(history, 1)
loss_plt(history, 2)
rate_loss_log(learning_rates, history, 3)


# ![Loss](graphs/model1/loss.png) ![Accuracy](graphs/model1/accuracy.png) ![Learning Rate](graphs/model1/learning_rate.png)

# In[106]:


history = compile_train(baseline2, 32, epochs, lr_start)
accuracy_plt(history, 4)
loss_plt(history, 5)
rate_loss_log(learning_rates, history, 6)


# ![Loss](graphs/model2/loss.png) ![Accuracy](graphs/model2/accuracy.png) ![Learning Rate](graphs/model2/learning_rate.png)

# In[ ]:


history = compile_train(baseline3, 32, epochs, lr_start)
accuracy_plt(history, 7)
loss_plt(history, 8)
rate_loss_log(learning_rates, history, 9)


# ![Loss](graphs/model3/loss.png) ![Accuracy](graphs/model3/accuracy.png) ![Learning Rate](graphs/model3/learning_rate.png)

# In[ ]:


history = compile_train(baseline4, 32, epochs, lr_start)
accuracy_plt(history, 10)
loss_plt(history, 11)
rate_loss_log(learning_rates, history, 12)


# ![Loss](graphs/model4/loss.png) ![Accuracy](graphs/model4/accuracy.png) ![Learning Rate](graphs/model4/learning_rate.png)

# In[105]:


def compile_train_fixed(model, batch_size, num_epochs, lr, checkpoint_path):

    opt = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()
    
    # tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # save weights callback
    checkpoint_path = checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr)
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, shuffle=True, callbacks=[cp_callback, callback]) #reduce_lr for Plateau approach
    
    return history


# # Choose learning rates from loss vs. log(learing rate) graphs above:
# 1. Model1: 15e-4
# 2.  Model2: 90e-5
# 3.  Model3: 20e-5
# 
# You can find the output of the training process here: "/outputs/model_training.txt"

# In[ ]:


history = compile_train_fixed(baseline1, 32, 300, 15e-4, "models/model1.cp")
accuracy_plt(history, 13)
loss_plt(history, 14)


# ![Loss](graphs/model1/loss_after.png) ![Accuracy](graphs/model1/accuracy_after.png)

# In[ ]:


history = compile_train_fixed(baseline2, 32, 300, 90e-5, "models/model2.cp")
accuracy_plt(history, 15)
loss_plt(history, 16)


# ![Loss](graphs/model2/loss_after.png) ![Accuracy](graphs/model2/accuracy_after.png)

# In[ ]:


history = compile_train_fixed(baseline3, 32, 300, 20e-5, "models/model3.cp")
accuracy_plt(history, 17)
loss_plt(history, 18)


# ![Loss](graphs/model3/loss_after.png) ![Accuracy](graphs/model3/accuracy_after.png)

# # Conclusions:
# 1. Model 1: We didn't hit the optimal learning rate for model 1. It might be hard to hit the optimal learning rate, because the slope of the drop in loss is very steep. We guess that there is a very specific optimal learning rate that is necessary to choose and we suggest to calculate it.
# 2. Model 2: The model starts to increase its performance signifcantly after 115 epochs and above. Nonetheless, the accuracy and loss are dropping a lot and the results doesn't seem to be stable.
# 3. Model 3: Two LSTMs are yielding satisfing results, since the accuracy and loss are improving very fast after about 20 epochs. Though, somtimes drops in accuracy and loss are observable, but overall it seems that the model is still learning afeter 200-300 epochs without divering training and test results. Thus, we estimate that this model is performing better than the others due to its stability and fast increase of the accuracy and decrease of the loss. 
# 4. So far, we suggest a model that is based on two LSTMs layers. We observed that an accuracy of 89 % can be reached and this kind of model still indicates potential for better learning. However, a fine-tuning of the parameters is still possible, but it's important to avoid overfitting.
# 5. We have been demonstrating that our approach to detect composite transposable elements based on gene frequency and protein families might be possible due to our binary classification approach. The binary classification indicates that the model can learn the pattern of gene frequency and protein families and that the signal are strong enough to detect composite transposable elements. We recommend to build a model for specific boundary predictions of composite transposable elements and to clean the input dataset towards confident and strong signals of composite transposable elements.
