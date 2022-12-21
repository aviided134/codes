import numpy as np 
np.random.seed(0)
x = [[0.1,-0.2,0.3,0.4],
	 [0.1,0.4,0.5,0.6],
	 [-0.7,0.5,0.2,-0.1]]

class layer_dense:
	def __init__(self, n_input, n_neuron):
		self.weight = 0.1*np.random.rand(n_input, n_neuron)
		self.bias = np.zeros((1, n_neuron))


	def forward(self, inputs):
		self.output = np.dot(inputs, self.weight) + self.bias

layer1 = layer_dense(4,5)
layer2 = layer_dense(5,2)


layer1.forward(x)
layer2.forward(layer1.output)

print(layer2.output)

