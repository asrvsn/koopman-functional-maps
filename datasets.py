import numpy as np

def delay_embed(X: np.ndarray, dt: int, d: int):
	'''
	Embed 1-dimensional time series into delay coordinates.
	dt: lag time 
	d: embedding dimension
	Returns d x n sample.
	'''
	if len(X.shape) > 1:
		raise Exception('input dimension too high')

	k = dt * d # need at least this much data
	n = X.shape[0] - k

	if n < 1:
		raise Exception('delay time and/or embedding dimension too high for sample size')

	Y = np.empty((n, d))
	for i in range(n):
		Y[i] = X[i:i+k:dt]
	return Y

def sinusoid_dataset(n: int, d: int):
	x = np.linspace(-n/10, n/10, n)
	y = np.sin(2*x)
	X = delay_embed(y, 1, d)
	return X