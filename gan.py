import tensorflow as tf
import numpy as np


class GAN:
    def __init__(self, input_size=784, random_size=100):
        self.input_size = input_size
        self.random_size = random_size

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def generator(self, z):
        # IMPLEMENTED THE GENERATOR USING THE G_ VARIABLES #
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        self.G_prob = tf.nn.sigmoid(G_log_prob)
        return self.G_prob

    def discriminator(self, x):
        # IMPLEMENTED THE DISCRIMINATOR USING THE D_ VARIABLES #
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def init_training(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.input_size])

        self.Z = tf.placeholder(tf.float32, shape=[None, self.random_size])

        self.G_W1 = tf.Variable(self.xavier_init([self.random_size, 128]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.G_W2 = tf.Variable(self.xavier_init([128, self.input_size]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.input_size]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self.D_W1 = tf.Variable(self.xavier_init([self.input_size, 128]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W2 = tf.Variable(self.xavier_init([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        self.G_sample = self.generator(self.Z)
        D_real, D_logit_real = self.discriminator(self.X)
        D_fake, D_logit_fake = self.discriminator(self.G_sample)

        # Implement the loss functions for training a GAN
        # -------------------
        self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(1.0 - D_fake))

        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)

    def generate_sample(self, num_samples):
        # GENERATE SAMPLES FROM THE GAN #
        samples = self.sess.run(self.G_sample, feed_dict={self.Z: self.sample_Z(num_samples, self.Z_dim)})
        return samples

    def train_model(self, data):

        mb_size = 128
        self.Z_dim = self.random_size

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for it in range(100000):
            X_mb, _ = data.train.next_batch(mb_size)

            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss],
                                           feed_dict={self.X: X_mb, self.Z: self.sample_Z(mb_size, self.Z_dim)})
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss],
                                           feed_dict={self.Z: self.sample_Z(mb_size, self.Z_dim)})
            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
