import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
# this is the first change in  the code
# this is second change in the code
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
									found_option = False				# if false, no reason to check from that class

								if found_option:
									opt_dict[np.linalg.norm(w_t)] = [w_t, b]    # ||w||: [w,b] 
									

				if w[0] < 0:
					optimized = True
					print('Optimized a step')
				else:
					w = w- step 


		norms = sorted([n for n in opt_dict])  # sorted according to the weight magnitude
		opt_choice = opt_dict[norms[0]]		   # optimum choice is the value([w,b]) of the least magnitude

		self.w = opt_choice[0]
		self.b = opt_choice[1]
		latest_optimum = opt_choice[0][0] + step * 2




	def predict(self, features):
		# sign wx + b
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		if classification != 0 and self.visualization:
			self.ax.scatter(features[0], features[1], s = 200, marker = *, c = self.colors[classification])  # s for size, c for color(used for classification) 
		
		return classification
		pass

	def visualize(self):
		[[self.ax.scatter(x[0], x[1], s = 200, c = self.colors[i]) for x in data_dict[i]] for i in data_dict] #  it will scatter the data according to the class. list comprehensin technique


		#the definition of hyperplane is x.w+b
		def hyperplane(x,w,b):

			#v = x*w+b
			#positive support vector(psv) = 1
			#nsv = 1
			#decision boundary = 0

			return(-w[0]*x-b+v) / w[1]


data_dict = {-1 : np.array([[1,7],
				    	   [2,8],
				   		   [3,8]]) ,
			
			  1 : np.array([[5,1],
			  	   			[6,-1],
			  	   			[7,3]]) }

