import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class Dataset():

    def __init__(self,states):

        self.train = Trainer(states)


    

class Trainer():

    def __init__(self,states):
        self.states = states

    def next_batch(self,size):

        data = self.states[np.random.randint(self.states.shape[0], size=size), :]
        
        return data, None

 