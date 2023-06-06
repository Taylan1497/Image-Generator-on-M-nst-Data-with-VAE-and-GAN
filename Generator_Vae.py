#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
#from model import model
import tensorflow as tf
from Model_Vae import model
tf.compat.v1.disable_eager_execution()

hidden_dim = 64

model = model()
_, _, train_mode, z, _, _ = model.encoder(hidden_dim)
decoder_output = model.decoder(z)

saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()

saver.restore(sess, './Model_Save/Model_VAE_100')
print("-> Checkpoints are restored from model")

randoms = [np.random.normal(0, 1, hidden_dim) for _ in range(100)]

print("-> Generation is started.")

images = sess.run(decoder_output, feed_dict = {z: randoms, train_mode: False})
images = [np.reshape(images[i], [28, 28]) for i in range(len(images))]

import matplotlib.pyplot as plt
def save_plot_2(examples,n):

  for i in range(n):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(examples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("Generated_images_try")
save_plot_2(images,16)
sess.close()


# In[ ]:




