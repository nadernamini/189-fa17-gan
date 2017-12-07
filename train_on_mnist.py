from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from kde import KDE
from gan import GAN


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_data = mnist.train.images[0:2000, :]
"""
# TRAIN KDE #
kde_model = KDE()
kde_model.train_model(train_data)
samples = kde_model.generate_sample(16)

fig = plot(samples)
plt.savefig('kde_mnist.png', bbox_inches='tight')
plt.close(fig)
"""

# TRAIN GAN #
gan_model = GAN()
gan_model.init_training()
gan_model.train_model(mnist)
samples = gan_model.generate_sample(16)

fig = plot(samples)
plt.savefig('gan_mnist.png', bbox_inches='tight')
plt.close(fig)
