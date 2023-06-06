#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
class model():

  def __init__(self):
    self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='x')
    self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='y')
    self.train_mode = tf.compat.v1.placeholder(tf.bool, name="train_mode")

  def encoder(self, hidden_dim):
    print("----- Encoder -----")
    print("input hidden_dim: ", hidden_dim)
    print("input x: ", self.x.shape)

    lstm, _ = tf.compat.v1.nn.dynamic_rnn(tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_dim), self.x, dtype=tf.float32)
    print("lstm: ", lstm.shape)
    flatten_lstm = tf.keras.layers.Flatten()(lstm)
    print("flatten_lstm: ", flatten_lstm.shape)

    mu = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu)(flatten_lstm)
    sigma = 0.5 * tf.keras.layers.Dense(units=hidden_dim)(flatten_lstm)
    epsilon = tf.random.normal(tf.stack([tf.shape(flatten_lstm)[0], hidden_dim]))
    z = mu + tf.multiply(epsilon, tf.exp(sigma))   
    print("mu: ", mu.shape)
    print("sigma: ", sigma.shape)
    print("epsilon: ", epsilon.shape)
    print("z: ", z.shape)

    return self.x, self.y, self.train_mode, z, mu, sigma


  def decoder(self, z):
    print("----- Decoder -----")
    print("input z: ", z.shape)
    reshaped_dimension = [-1, 14, 14, 1]

    fc1 = tf.keras.layers.Dense(units=98, activation=tf.nn.relu)(z)
    print("1. fc: ", fc1.shape)
    fc2 = tf.keras.layers.Dense(units=196, activation=tf.nn.relu)(fc1)
    print("2. fc: ", fc2.shape)

    reshaped_fc2 = tf.reshape(fc2, reshaped_dimension)
    print("2. reshaped_fc: ", reshaped_fc2.shape)

    tconv1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(reshaped_fc2)
    dropout1 = tf.keras.layers.Dropout(rate=0.75)(tconv1)
    print("3. tconv: ", tconv1.shape)
    print("3. dropout: ", dropout1.shape)

    tconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=0.75)(tconv2)
    print("4. tconv: ", tconv2.shape)
    print("4. dropout: ", dropout2.shape)

    tconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(dropout2)
    #dropout3 = tf.layers.dropout(tconv3, rate=0.75, training=self.train_mode)
    print("5. tconv: ", tconv3.shape)
    #print("5. dropout: ", dropout3.shape)

    flatten_tconv3 = tf.keras.layers.Flatten()(tconv3)
    print("5. flatten_tconv: ", flatten_tconv3.shape)

    fc3 = tf.keras.layers.Dense(units=28*28, activation=tf.nn.sigmoid)(flatten_tconv3)
    print("6. fc: ", fc3.shape)

    reshaped_output = tf.reshape(fc3, shape=[-1, 28, 28])
    print("6. reshaped_output: ", reshaped_output.shape)

    return reshaped_output


  def loss(self, decoder, mu, sigma):
    flatten_y = tf.reshape(self.y, shape=[-1, 28 * 28])
    unreshaped = tf.reshape(decoder, [-1, 28 * 28])
    unreshaped = tf.clip_by_value(unreshaped, 1e-7, 1-1e-7) # clipping is added for bce loss problems

    # binary cross entropy loss (reconstruction loss)
    bce_loss = tf.reduce_sum(-flatten_y*tf.math.log(unreshaped)-(1.0 - flatten_y)*tf.math.log(1.0 - unreshaped), axis=1)

    # kl divergence loss:
    kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sigma - tf.square(mu) - tf.exp(2.0 * sigma), axis=1)

    # general loss
    general_loss = tf.reduce_mean(bce_loss + kl_loss, axis=0)

    return general_loss, bce_loss, kl_loss


  def optimizer(self, loss, learning_rate):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer


# In[ ]:




