from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, concatenate 
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import Callback

from tensorflow.python.ops import nn
import tensorflow as tf

import numpy as np
import argparse
import os
import random
from math import log

from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score 

import sys

# dataset
movies = np.load("npy/movies.npy")
books = np.load("npy/books.npy")

#content
np_content = np.load("npy/docvec_merged1024.npy")
index = int(np_content.shape[0]/2)
print("xxxxxxxxxx", index)
c_books = np_content[:index,:]
c_movies = np_content[index:,:]

sys.stdout.flush()

test_users = 500
test_movie = 20
test_book = 10

def unison_shuffled_copies(a, b, c, d):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

movies, books, c_books, c_movies = unison_shuffled_copies(movies, books, c_books, c_movies)

movies1 = movies
books1 = books
c_movies1 = c_movies
c_books1 = c_books

movies2 = np.empty((0, movies1.shape[1]))  
books2 = np.empty((0, books1.shape[1]))  
c_movies2 = np.empty((0, c_movies1.shape[1]))  
c_books2 = np.empty((0, c_books1.shape[1]))  

r = random.SystemRandom()

comp_keys = list()

candidates_m = np.where(np.sum(movies1, axis=1) >= test_movie)[0]
candidates_b = np.where(np.sum(books1, axis=1) >= test_book)[0]
candidates_cond = np.intersect1d(candidates_m, candidates_b)

print("candidate length", len(candidates_cond))
sys.stdout.flush()

np.random.shuffle(candidates_cond)

for randint in candidates_cond:

    movies2 = np.insert(movies2, movies2.shape[0], movies1[randint],  axis=0) 
    books2 = np.insert(books2, books2.shape[0], books1[randint],  axis=0) 
    c_movies2 = np.insert(c_movies2, c_movies2.shape[0], c_movies1[randint],  axis=0) 
    c_books2 = np.insert(c_books2, c_books2.shape[0], c_books1[randint],  axis=0) 

    comp_keys.append(randint)

    if len(movies2) == test_users:
        break

    print ("length of test users", len(movies2))
    sys.stdout.flush()

movies1 = np.delete(movies1, comp_keys, axis=0)
books1 = np.delete(books1, comp_keys, axis=0)
c_movies1 = np.delete(c_movies1, comp_keys, axis=0)
c_books1 = np.delete(c_books1, comp_keys, axis=0)

# network parameters
original_dim = movies.shape[1]
original_dim2 = books.shape[1] 
content_dim = c_movies.shape[1]
content_dim2 = 256 

layer1_dim = 512
layer2_dim = 256 
latent_dim = 128    

print("data generation done")
print("network", original_dim, original_dim2, layer1_dim, layer2_dim, latent_dim, content_dim, content_dim2)

sys.stdout.flush()

batch_size = 128    
epochs = 50
decay = 1e-4 
bias = True
alpha = 0
beta = 1
gama = 1
hadamard = 30  

ecount = 0

print("HADAMARD", hadamard)

test_ratio = 4
print("test_ratio", test_ratio)

print("updated 1 step") 
print("tanh", "adam")
print("params", batch_size, epochs, decay, alpha, beta, gama, hadamard)

print("books: ", books.shape[0], books.shape[1])
print("movie: ", movies.shape[0], movies.shape[1])
print("books non zero:", np.count_nonzero(books))

print("books1: ", books1.shape[0], books1.shape[1])
print("books2: ", books2.shape[0], books2.shape[1])
print("movie1: ", movies1.shape[0], movies1.shape[1])
print("movie2: ", movies2.shape[0], movies2.shape[1])

print("c books1: ", c_books1.shape[0], c_books1.shape[1])
print("c books2: ", c_books2.shape[0], c_books2.shape[1])
print("c movie1: ", c_movies1.shape[0], c_movies1.shape[1])
print("c movie2: ", c_movies2.shape[0], c_movies2.shape[1])

sys.stdout.flush()

r = random.SystemRandom()

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

#neural nets
input_shape = (original_dim, )
input_shape2 = (original_dim2, )
input_shape_cont = (content_dim, )
input_shape_cont2 = (content_dim, )

# VAE model = encoder + decoder
# build encoder model
#for s
inputs_s = Input(shape=input_shape, name='encoder_s1_input')
inputs_s_cont = Input(shape=input_shape_cont , name='content_input')

se1 = Dense(layer1_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(inputs_s)
se2 = Dense(layer2_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(se1)

sec = Dense(content_dim2, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(inputs_s_cont)

secom = concatenate([se2, sec])

z_mean_s = Dense(latent_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, name='z_mean_s')(secom)
z_log_var_s = Dense(latent_dim,  kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, name='z_log_var_s')(secom)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z_s = Lambda(sampling, output_shape=(latent_dim,), name='z_s')([z_mean_s, z_log_var_s])

z_si = Dense(latent_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh', name='z_si')(z_s)

#for i
inputs_i = Input(shape=input_shape2, name='encoder_i_input')
inputs_i_cont = Input(shape=input_shape_cont2, name='content_input2')

ie1 = Dense(layer1_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(inputs_i)
ie2 = Dense(layer2_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(ie1)

iec = Dense(content_dim2, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(inputs_i_cont)

iecom = concatenate([ie2, iec])

z_mean_i = Dense(latent_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, name='z_mean_i')(iecom)
z_log_var_i = Dense(latent_dim,  kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, name='z_log_var_i')(iecom)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z_i = Lambda(sampling, output_shape=(latent_dim,), name='z_i')([z_mean_i, z_log_var_i])

# instantiate encoder model
print("encoder models")
encoder_s = Model([inputs_s, inputs_s_cont], [z_mean_s, z_log_var_s, z_s, z_si], name='encoder_s')
encoder_s.summary()
print("encoder models")
encoder_i = Model([inputs_i, inputs_i_cont], [z_mean_i, z_log_var_i, z_i], name='encoder_i')
encoder_i.summary()

# build decoder model
latent_inputs_s = Input(shape=(latent_dim,), name='z_samplings')

#sdc = Dense(content_dim2, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(latent_inputs_s)
#sdcont = Dense(content_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias)(sdc)

sd2 = Dense(layer2_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(latent_inputs_s)
sd1 = Dense(layer1_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(sd2)
outputs_s = Dense(original_dim, activation='sigmoid')(sd1)

latent_inputs_i = Input(shape=(latent_dim,), name='z_samplingi')

#idc = Dense(content_dim2, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(latent_inputs_i)
#idcont = Dense(content_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias)(idc)

id2 = Dense(layer2_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(latent_inputs_i)
id1 = Dense(layer1_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=bias, activation='tanh')(id2)
outputs_i = Dense(original_dim2, activation='sigmoid')(id1)

# instantiate decoder model
print("decoder models")
#decoder_s = Model(latent_inputs_s, [outputs_s, sdcont], name='decoder_s')
decoder_s = Model(latent_inputs_s, [outputs_s], name='decoder_s')
decoder_s.summary()

print("decoder models")
#decoder_i = Model(latent_inputs_i, [outputs_i, idcont], name='decoder_i')
decoder_i = Model(latent_inputs_i, [outputs_i], name='decoder_i')
decoder_i.summary()

# instantiate VAE model
#outputs_s, output_s_cont = decoder_s(encoder_s([inputs_s, inputs_s_cont])[2])
outputs_s = decoder_s(encoder_s([inputs_s, inputs_s_cont])[2])
#outputs_i, output_i_cont = decoder_i(encoder_s([inputs_s, inputs_s_cont])[3])
outputs_i = decoder_i(encoder_s([inputs_s, inputs_s_cont])[3])

print("final models")
#vae = Model([inputs_s, inputs_i, inputs_s_cont, inputs_i_cont], [outputs_s, outputs_i, output_s_cont, output_i_cont], name='vae_mlp')
vae = Model([inputs_s, inputs_i, inputs_s_cont, inputs_i_cont], [outputs_s, outputs_i], name='vae_mlp')
vae.summary()

print("prediction models")
output_mod = Model([inputs_s, inputs_s_cont], outputs_i, name='output_model')
output_mod.summary()

sys.stdout.flush()

class Histories(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("predictions starting")
        predictions = output_mod.predict([movies2, c_movies2], batch_size=batch_size)
        print("predictions done")
        sys.stdout.flush()

        positives = 0
        negatives = 0
        user_ranks = list()

        for user in range(movies2.shape[0]):

            movieonevals = np.where(movies2[user] == 1)[0]
            prediction = predictions[user]
 
            onevals = np.where(books2[user] == 1)[0]
            zerovals = np.where(books2[user] == 0)[0]

            rusers = []
            comp_values = list()

            #select random user
            while True:
                ruser = r.randint(0, (movies2.shape[0] - 1))

                if ruser == user:
                    continue 

                ronevals = np.where(books2[ruser] == 1)[0]

                rusers.append(ruser)

                if len(rusers) == test_ratio:
                    break

            user_value = 0
            ruser_final = 100

            for val in onevals:
                user_value += (1-prediction[val])**3

            user_final = user_value/len(onevals) 

            for ruser in rusers:
                ronevals = np.where(books2[ruser] == 1)[0]
                #rzerovals = np.where(books2[ruser] == 0)[0]
            
                ruser_value_t = 0
                for val in ronevals:
                    ruser_value_t += (1-prediction[val])**3

                ruser_value_t = ruser_value_t/len(ronevals) 
                comp_values.append(ruser_value_t)

                if ruser_value_t < ruser_final:
                    ruser_final = ruser_value_t 

            if user_final < ruser_final: 
                positives += 1
            else: 
                negatives += 1

            rank = [i for i in comp_values if i < user_final] 
            rank = len(rank) #+ 1
            user_ranks.append(rank)

        evranks = [2, 3, 4, 5, 10, 20, 50]        
        for rank in evranks:
        
            HR = (sum(vv <  rank for vv in user_ranks)/float(len(user_ranks)))

            NDCG = 0.0
            for val in user_ranks:
                if val < rank:
                    NDCG += log(2) / log(val + 2)
            NDCG = NDCG/float(len(user_ranks))

            print(epoch, "user results xxx rank HR NDCG", positives, negatives, positives/(positives+negatives), "xxx", rank, HR, NDCG)

        sys.stdout.flush()
        return

def custom_crossentropy(inputs, outputs, hadamard):
    #for sig-moid only
    e1 = K.mean(K.binary_crossentropy(inputs, outputs), axis=-1) 
    outputs = outputs*inputs
    e2 = K.mean(K.binary_crossentropy(inputs, outputs), axis=-1) 
    return (e1 + hadamard*e2) 

def hadamard_squared_error(y_true, y_pred):
    print("hadamard value:", hadamard)
    B = y_true * (hadamard - 1) + 1
    return K.mean(tf.multiply(K.square(y_pred - y_true), B), axis=-1)

if __name__ == '__main__':

    reconstruction_loss_s = custom_crossentropy(inputs_s, outputs_s, hadamard)
    reconstruction_loss_s *= original_dim

    reconstruction_loss_i = custom_crossentropy(inputs_i, outputs_i, hadamard)
    reconstruction_loss_i *= original_dim2

    #reconstruction_loss_s_cont = mse(output_s_cont, inputs_s_cont)
    #reconstruction_loss_s_cont *= content_dim

    #reconstruction_loss_i_cont = mse(output_i_cont, inputs_i_cont)
    #reconstruction_loss_i_cont *= content_dim

    kl_loss_s = 1 + z_log_var_s - K.square(z_mean_s) - K.exp(z_log_var_s)
    kl_loss_s = K.sum(kl_loss_s, axis=-1)
    kl_loss_s *= -0.5

    kl_loss_i = 1 + z_log_var_i - K.square(z_mean_i) - K.exp(z_log_var_i)
    kl_loss_i = K.sum(kl_loss_i, axis=-1)
    kl_loss_i *= -0.5

    custom_loss = mse(z_si, z_i)
    custom_loss *= latent_dim 
 
    #vae_loss = K.mean((reconstruction_loss_s + reconstruction_loss_i + reconstruction_loss_s_cont + reconstruction_loss_i_cont) + (kl_loss_s + kl_loss_i) + custom_loss)
    vae_loss = K.mean((reconstruction_loss_s + reconstruction_loss_i) + (kl_loss_s + kl_loss_i) + custom_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    # prepare callback
    histories = Histories()

    vae.fit([movies1, books1, c_movies1, c_books1], epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[histories])


