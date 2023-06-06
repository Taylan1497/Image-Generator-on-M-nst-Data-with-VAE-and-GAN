#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tensorflow / Keras
from skimage.transform import resize
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from numpy import vstack
from keras import Input
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from matplotlib import pyplot
from numpy import expand_dims
from keras import backend 
#print('Tensorflow/Keras: %s' % keras.__version__)
from keras.models import Model
from keras.datasets.mnist import load_data
# prepare the inception v3 model
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd


# In[2]:


# Load digits data 
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Print shapes
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)

# Normalize input data (divide by 255) 
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255





# initial parameters
learning_rate = 0.001
batch_size = 256
epoch_number = 100
hidden_dim = 64




#### FID CALCULATION ####
def scale_images(images, new_shape=(299,299,3)):
    images_list = list()
    for image in images:
 # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
 # store
        images_list.append(new_image)
    return asarray(images_list)
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid




from Model_Vae import model
# vae model
tf.compat.v1.disable_eager_execution()
model = model()
x, y, train_mode, z, mu, sigma = model.encoder(hidden_dim)
decoder_output = model.decoder(z)
general_loss, rec_loss, kl_loss = model.loss(decoder_output, mu, sigma)
train_optimizer = model.optimizer(general_loss, learning_rate)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print("---Session Started.----")



#SET FID CALCULATION 

Calculate_fid =False
if Calculate_fid==True:
    model_fid = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    print("FID calculation activated for each 10 epoch.")
if Calculate_fid==False:
    print("FID calculation deactivated.")
# main training loop
plot_general_loss_, plot_rec_loss_, plot_kl_loss_, FID_epoch = [], [], [],[]

for epoch in range(epoch_number):

    number_of_batch = int(len(X_train) / batch_size)

    general_loss_, rec_loss_, kl_loss_,fid_batch = 0, 0, 0, 0

    for i in range(number_of_batch):

        batch = X_train[i*batch_size:(i+1)*batch_size]  # we take only data (not labels)
        batch = np.reshape(batch, [-1, 28, 28])  # reshape the data

        # train the network for one batch
        _, general_loss_, rec_loss_, kl_loss_, decoder_ = sess.run(
            [train_optimizer, general_loss, rec_loss, kl_loss, decoder_output],
            feed_dict={x: batch, y: batch, train_mode: True})
        
        if i == number_of_batch-1 and epoch%10==0 and Calculate_fid==True: # For FID values
            randoms = [np.random.normal(0, 1, hidden_dim) for _ in range(100)]
            decoder_images = sess.run(decoder_output, feed_dict = {z: randoms, train_mode: False})
            #plt.imshow[decoder_images[0]]
            images1=scale_images(batch,new_shape=(299,299,3))
            images2=scale_images(decoder_images,new_shape=(299,299,3))
            fid_batch = calculate_fid(model_fid,images1,images2)
            print("Fid Value:",fid_batch)
            FID_epoch.append(fid_batch)
        if i == 0: # at the end of the each epoch, or at the beginning.
            print(
                "|--- Epoch: {} \t---> General loss: {:.2f} | Reconstruction loss: {:.2f} | KL-divergence loss: {:.2f}".format(
                    epoch, general_loss_, np.mean(rec_loss_), np.mean(kl_loss_)))

            plot_general_loss_.append(general_loss_)
            plot_rec_loss_.append(np.mean(rec_loss_))
            plot_kl_loss_.append(np.mean(kl_loss_))
            #FID_epoch.append(fid_batch)


# In[8]:


def plot_loss(plot_general_loss_,plot_rec_loss_,plot_kl_loss_,epochs):
  loss_frame = pd.DataFrame({"Total Loss":plot_general_loss_,"Reconstruction Loss":plot_rec_loss_,"KL-Divergence Loss":plot_kl_loss_})
  plot = loss_frame.plot(xlabel="Epochs",ylabel="Loss")
  plt.figure(figsize=(10, 6))
  plt.savefig("plot_loss_VAE_30_May")
  return plot


# In[ ]:


epochs=100
plot_loss(plot_general_loss_,plot_rec_loss_,plot_kl_loss_,epochs)


# In[9]:


def plot_fid(fid_values,epoch=100):
  epoch_list=[]
  for i in range(epoch):
    if i%10==0:
      epoch_list.append(i)

  ax = plt.figure(figsize=(10,6))
  #gs = gridspec.GridSpec(2, 2)
  plot = plt.plot(epoch_list,fid_values)

  plt.ylabel('FID Value')
  plt.xlabel('Epochs')
  plt.savefig("Fid_Plot")

  return plot


# In[ ]:


if Calculate_fid==True:
    plot_fid(FID_epoch,epoch=100)
    print("FID Plot Created")


# In[ ]:


# save model
saver = tf.train.Saver()
saver.save(sess, "model")
print("-> Model is saved.")

sess.close()

print("-> Session is ended.")

