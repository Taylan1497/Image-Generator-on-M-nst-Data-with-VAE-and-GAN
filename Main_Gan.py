#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Tensorflow / Keras
from skimage.transform import resize
import numpy
import tensorflow as tf
from tensorflow import keras
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from numpy import vstack
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
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
from Model_Gan import discriminator, generator, GAN
# load the images 
(trainX, trainy), (testX, testy) = load_data()
# The Shape of train and test set
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

# implementation of wasserstein loss
import numpy as np

def wasserstein_loss(y_true, y_pred):
 return backend.mean(y_true) * backend.mean(y_pred)
# prepare the inception v3 model
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

Calculate_Fid=False
if Calculate_Fid:
  model_fid = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
  print("FID Calculation Activated")
if Calculate_Fid==False:
  print("FID calculation deactivated.")

# scale an array of images to a new size
def scale_images(images, new_shape):
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



# load and prepare mnist training images
def load_real_samples():
    # load mnist dataset
    (trainX, _), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255.0
    return X

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y




# In[ ]:


# example of defining and using the generator model
from numpy import zeros
from numpy.random import randn
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y


# In[ ]:


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = discriminator()
# create the generator
g_model = generator(latent_dim)
# create the gan
gan_model = GAN(g_model, d_model)


# In[ ]:


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    d_loss_epoch=[]
    g_loss_epoch=[]
    FID_epoch=[]
    # manually enumerate epochs
    for i in range(n_epochs):
        epoch_d_loss=0
        epoch_g_loss=0
        fid_epoch=0
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            #Calculate FID
            if j==int(dataset.shape[0]/n_batch)-1 and i%10==0 and Calculate_Fid==True: # at the end of the batch and for each 10 epoch
              X_real_fid=scale_images(X_real,(299,299,3))
              X_fake_fid = scale_images(X_fake,(299,299,3))
              fid_batch = calculate_fid(model_fid,X_real_fid,X_fake_fid)
              print("FID VALUE:", fid_batch, "Epoch:",i)
              FID_epoch.append(fid_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            
            epoch_d_loss += d_loss
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            epoch_g_loss += g_loss
            # summarize loss on this batch
        epoch_d_loss /= bat_per_epo
        epoch_g_loss /= bat_per_epo
        #fid_epoch/=bat_per_epo
        #print("FID VALUE:", fid_epoch)
        print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, epoch_d_loss, epoch_g_loss))
        d_loss_epoch.append(epoch_d_loss)
        g_loss_epoch.append(epoch_g_loss)
        #FID_epoch.append(fid_batch)
    return d_loss_epoch, g_loss_epoch, FID_epoch


# In[ ]:


from skimage.transform import resize
import numpy
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = discriminator()
# create the generator
g_model = generator(latent_dim)
# create the gan
gan_model = GAN(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
d_loss_epoch, g_loss_epoch,FID_epoch=train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256)


# In[ ]:


#pyplot.savefig("plot_loss")
import matplotlib.pyplot as plt
import pandas as pd
def plot_loss(d_loss_epoch,g_loss_epoch,epochs=100):
  loss_frame = pd.DataFrame({"D_loss":d_loss_epoch,"G_loss":g_loss_epoch})
  plot = loss_frame.plot(xlabel="Epoch",ylabel="G and D Loss")
  plt.figure(figsize=(10, 6))
  plt.savefig("plot_loss_Gan")
  return plot
plot_loss(d_loss_epoch,g_loss_epoch,epochs=100)


# In[ ]:


def plot_fid(fid_values,epoch=20):
  epoch_list=[]
  for i in range(epoch):
    if i%10==0:
      epoch_list.append(i)
  #epoch_list = [i if i+1%10==0 else continue for i in range(epoch)]
  ax = plt.figure(figsize=(10,6))
  #gs = gridspec.GridSpec(2, 2)
  plot = plt.plot(epoch_list,fid_values)
  #ax = fig.add_subplot()
  #ax.plot(fid_values,label="FID Values",)
  plt.ylabel('FID Value')
  plt.xlabel('Epochs')
  plt.savefig("Fid_Plot_Gan")

  return plot
if Calculate_Fid==True:
    plot_fid(FID_epoch,epoch=100)


# In[ ]:


import pickle
pickle.dump(g_model, open('Model_Gan.pkl', 'wb'))
print("Model is saved.")


# In[ ]:





# In[ ]:





# In[ ]:




