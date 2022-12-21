import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style


'''
__init__ method is called everytime whenever a class is called or an object is created from a cass.
in this class, init method initializes two attributes: visualization and colors
to acces these attribues, we can use dot(.) notation. like 
'''

#When you define a class in Python, every method that you define, must accept that instance as its first argument (called self by convention).
class Support_Vector_Machine: 
	def __init__(self, visualization = True):  # visualization is to see the graphs and all. _init__, fit and predict are methods of a class
		self.visualization = visualization
		self.colors = {1 : 'r', -1 : 'b'}
		if self.visualization:
			self.fig = plt.fig()
			self.ax = self.fig.add_subplot(1,1,1)

	def fit(self, data):
		self.data = data 

		opt_dict = {}
		transform = [[1,1],
					 [-1,1],
					 [-1,-1]
					 [1,-1]]

		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for features in featureset:
					all_data.append(features)
		
		self.max_feature_data = max(all_data)
		self.min_feature_data = min(all_data)	
		all_data = None 

		step_sizes = [self.max_feature_data * 0.1 ,   #big step sizes. otherwise it takes alot of time
					 self.max_feature_data * 0.01, 	 # smaller step sizes gradually
					 self.max_feature_data * 0.001]

		b_range_multiple = 5

		# we donot need to take small steps, extremely expensive if multiple not provided
		b_multiple = 5

		latest_optimum = self.max_feature_data * 10   #intial value is high, because we test the weight verctor from higher to lower

		for step in step_sizes:
			w = np.array([latest_optimum, latest_optimum])

			optimized = False
			while not optimized:
				# in this for loop, a variable is created to store all the numbers from highest point * b_range_multiplier to lower point * b_range_multiplier with the step
				
				for b in np.arange((-1 * (self.max_feature_data) * b_range_multiple),	#the arange(staring point, ending point, step)
								   (+1 * (self.max_feature_data) * b_range_multiple),   
								   (step * b_multiple)):

					for transformations in transform:
						w_t = w * transformations
						found_option = True

						for i in self.data:  #i is class(-1,1)
							for xi in self.data[i]:
								yi = i 
								if not yi*(np.dot(w_t, xi) + b) >= 1:
									found_option = False

								if found_option:
									






	def predict(self, features):
		# sign wx + b
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		return classification
		pass

data_dict = {-1 : np.array([[1,7],
				    	   [2,8],
				   		   [3,8]]) ,
			
			  1 : np.array([[5,1],
			  	   			[6,-1],
			  	   			[7,3]]) }

