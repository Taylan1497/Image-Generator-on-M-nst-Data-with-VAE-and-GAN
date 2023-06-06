#!/usr/bin/env python
# coding: utf-8

# In[3]:


# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import pickle
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
 # generate points in the latent space
 x_input = randn(latent_dim * n_samples)
 # reshape into a batch of inputs for the network
 x_input = x_input.reshape(n_samples, latent_dim)
 return x_input


# In[4]:


pickled_model = pickle.load(open('./Model_Save/Model_Gan.pkl', 'rb'))


# In[6]:


import matplotlib.pyplot as plt
def save_plot_2(examples, n):

  for i in range(n):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(examples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("Generated_images_wasser_GAN_Final")


# In[7]:


#pickled_model = pickle.load(open('_generator_model_GAN.pkl', 'rb'))
# generate images
latent_points = generate_latent_points(100, 100)
# generate images
X = pickled_model.predict(latent_points)
# plot the result
save_plot_2(X,100)


# In[ ]:




