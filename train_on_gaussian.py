import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from dataset import Dataset

from numpy.random import normal

from kde import KDE
from gan import GAN

def plot(ground_truth_samples,generated_samples):
    #IPython.embed()

    bins = np.linspace(-3.5, 3.5, 100)
    plt.hist(ground_truth_samples,bins)
    plt.hist(generated_samples,bins)

    IPython.embed()
    
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    print 'got here'

    fig = plt.gcf()



    return fig

N_SAMPLES = 1000

train_data = normal(size=(1,N_SAMPLES))
train_data = train_data.T



####TRAIN KDE####v##
kde_model = KDE(use_pca=False)
kde_model.train_model(train_data)
samples = kde_model.generate_sample(N_SAMPLES)

fig = plot(train_data,samples)
plt.savefig('gaussian_kde.png', bbox_inches='tight')
plt.close(fig)


#####TRAIN GAN#######
gan_model = GAN(input_size = 1, random_size = 1)
gan_model.init_training()

#Create dataset
train_data_m = Dataset(train_data)
gan_model.train_model(train_data_m)
samples = gan_model.generate_sample(N_SAMPLES)


fig = plot(train_data,samples)
plt.savefig('gaussian_gan.png', bbox_inches='tight')
plt.close(fig)



