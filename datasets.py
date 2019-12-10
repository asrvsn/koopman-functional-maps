import numpy as np

from preprocess import delay_embed

def sinusoid_dataset(n: int, d: int):
	'''
	1-dimensional sin() dataset
	'''
	x = np.linspace(-n/10, n/10, n)
	y = np.sin(2*x)
	X = delay_embed(y, 1, d)
	return X