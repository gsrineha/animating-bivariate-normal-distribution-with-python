#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# from jax.scipy.stats import multivariate_normal


def plot_pdf(cov_val=[-0.8, 0, 0.8], mean=np.array([0, 0])):
	plt.style.use('seaborn-dark')
	plt.rcParams['figure.figsize']=14,6
	fig = plt.figure()

	# Initializing the random seed
	random_seed=1000

	# List containing the variance
	# covariance values
	# cov_val = [-0.8, 0, 0.8]

	# Setting mean of the distributino
	# to be at (0,0)
	mean = np.array([0,0])

	# Storing density function values for
	# further analysis
	pdf_list = []

	# Iterating over different covariance values
	for idx, val in enumerate(cov_val):
		
		# Initializing the covariance matrix
		cov = np.array([[1, val], [val, 1]])
		
		# Generating a Gaussian bivariate distribution
		# with given mean and covariance matrix
		distr = multivariate_normal(cov = cov, mean = mean,
									seed = random_seed)
		
		# Generating a meshgrid complacent with
		# the 3-sigma boundary
		mean_1, mean_2 = mean[0], mean[1]
		sigma_1, sigma_2 = cov[0,0], cov[1,1]
		
		x = np.linspace(-3*sigma_1, 3*sigma_1, num=100)
		y = np.linspace(-3*sigma_2, 3*sigma_2, num=100)
		X, Y = np.meshgrid(x,y)
		
		# Generating the density function
		# for each point in the meshgrid
		pdf = np.zeros(X.shape)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				# print(i, j, X[i, j])
				pdf[i,j] = multivariate_normal.pdf([X[i,j]], mean=mean, cov=cov) # distr.pdf([X[i,j], Y[i,j]])

		# Plotting the density function values
		key = 131+idx
		ax = fig.add_subplot(key, projection = '3d')
		ax.plot_surface(X, Y, pdf, cmap = 'viridis')
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.title(f'Covariance between x1 and x2 = {val}')
		pdf_list.append(pdf)
		ax.axes.zaxis.set_ticks([])

	plt.tight_layout()
	plt.show()
 
plot_pdf()


# In[52]:


get_ipython().run_line_magic('matplotlib', '--list')
get_ipython().system('pip install -q ipympl')
import ipywidgets as widgets
widgets.interact(plot_pdf, cov_val=[[-0.6, 0, 0.6], [-0.8, 0, 0.8], [-0.1, 0, 0.1]]);

