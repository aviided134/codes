import numpy as np 

data_dict = {-1 : np.array([[1,7],
				    	   [2,8],
				   		   [3,8]]) ,
			
			  1 : np.array([[5,1],
			  	   			[6,-1],
			  	   			[7,3]]) }

all_data =[]
a = 0
for yi in data_dict:
	#print(yi)
	#a = a +1
	for featureset in data_dict[yi]:
		print(featureset)	
		#for feature in featureset:
			#all_data.append(feature)

latest = 10000

step_sizes = [100,10,1]		
for step in step_sizes:
	w = np.array([latest, latest])
	print(w)
	#for b in np.arange(-100*5 , 100*5 , step * 5 ):

		#print(b)

