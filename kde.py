import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



class KDE():


	def __init__(self,use_pca = True):

		self.use_pca = use_pca


	def train_model(self,data,pca=True):
		# project the 64-dimensional data to a lower dimension

		if self.use_pca:
			self.pca = PCA(n_components=15, whiten=False)
			
			data = self.pca.fit_transform(data)

		# use grid search cross-validation to optimize the bandwidth
		params = {'bandwidth': np.logspace(-1, 1, 20)}
		

		###FILL IN KDE FITTING AND GRIDSEARCH OPTIMIZATION


		

	def generate_sample(self,K):
		###GENERATE SAMPLES FROM KDE

		if self.use_pca:
			new_data = self.pca.inverse_transform(new_data)
		
		
		return new_data



