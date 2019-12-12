import numpy as np
import random
import pywt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def block_bootstrap_example(X: np.ndarray, block_size: int, offset: int):
	'''
	Samples X_t by block bootstrap.
	X: numpy tensor (m channels x n time steps)
	block_size: size of each block
	offset: target offset
	'''
	n = X.shape[0]
	t = random.randint(0, n-block_size-offset)
	X1, X2 = X[t:t+block_size], X[t+offset:t+block_size+offset]
	X1, X2 = np.transpose(X1, (1, 0, 2)), np.transpose(X2, (1, 0, 2))
	return X1, X2

def delay_embed_1d(X: np.ndarray, lag: int, d: int):
	'''
	Embed time series into delay coordinates. Assumes first dimension is time.
	lag: lag time 
	d: embedding dimension
	Returns d x n sample. Last dimension will be delay coordinates.
	'''

	k = lag * d # need at least this much data
	n = X.shape[0] - k

	if n < 1:
		raise Exception('delay time and/or embedding dimension too high for sample size')

	Y = np.empty((n, d))
	for i in range(n):
		Y[i] = X[i:i+k:lag]
	return Y

def delay_embed_2d(X: np.ndarray, lag: int, d: int):
	'''
	Embed time series into delay coordinates. Assumes first dimension is time.
	lag: lag time 
	d: embedding dimension
	Returns d x n sample. Last dimension will be delay coordinates.
	'''

	k = lag * d # need at least this much data
	n = X.shape[0] - k

	if n < 1:
		raise Exception('delay time and/or embedding dimension too high for sample size')

	Y = np.empty((n, 2, d))
	for i in range(n):
		Y[i][0] = X[i:i+k:lag][:,0]
		Y[i][1] = X[i:i+k:lag][:,1]
	return Y

def un_embed_2d(X: np.ndarray, lag: int):
	x1 = X[:,0,0]
	x2 = X[:,1,0]
	return x1, x2

def coefs_fourier(X_t: np.ndarray):
	'''
	Compute fourier coefficients for sample.
	X_t: sample at a particular time
	d: number of kernel parameters to try 
	'''
	return np.fft.rfftn(X_t)

def inverse_fourier(C_t: np.ndarray):
	return np.fft.irfftn(C_t)

def coefs_wavelet(X_t: np.ndarray, wavelet: pywt.Wavelet):
	'''
	Compute wavelet coefficients for sample.
	X_t: sample at a particular time
	wavelet: pywt.Wavelet()
	'''
	if len(X_t.shape) > 2:
		# TODO
		raise Exception('Cant process more than 1D samples')
	cA, (cH, cV, cD) = pywt.dwt2(X_t, wavelet, mode='smooth')
	return np.block([[cA, cH], [cV, cD]])

def inverse_wavelet(C_t: np.ndarray, wavelet: pywt.Wavelet):
	(m, n) = C_t.shape
	h, v = int(m/2), int(n/2)
	cA, cH, cV, cD = C_t[:h, :v], C_t[:h, v:], C_t[h:, :v], C_t[h:, v:]		
	return pywt.idwt2((cA, (cH, cV, cD)), wavelet, mode='smooth')

class TimeSeriesDataset(Dataset):
	def __init__(self, X: np.ndarray, coefs_fn, offset=1, block_size=None, n_samples=None, device=torch.device("cpu"), normalize=None):
		'''
		Data loader for time series.
		Uses nxm indexing convention because numpy does.
		'''
		n, d, m = X.shape # TODO multi-dimensional samples
		if block_size is None:
			block_size = m
		if n_samples is None:
			n_samples = int(n / block_size)

		X_t_sample, _ = block_bootstrap_example(X, block_size, offset)

		self.X = X
		self.n = n_samples
		self.device = device

		self.samples = np.empty((n_samples, 2, d, block_size, m))
		self.samples_C = np.empty((n_samples, 2, d, block_size, m))
		for i in range(n_samples):
			X1, X2 = block_bootstrap_example(X, block_size, offset)
			C1, C2 = X1, X2
			self.samples[i][0], self.samples[i][1] = X1, X2
			self.samples_C[i][0], self.samples_C[i][1] = C1, C2

		self.normalize = normalize
		if normalize is None:
			self.norm_x = 1
			self.norm_c = 1
		else:
			self.norm_x = np.abs(self.samples).max() * float(normalize)
			self.norm_c = np.abs(self.samples_C).max() * float(normalize)

	def __len__(self):
		return self.n

	def __getitem__(self, idx):
		X = torch.from_numpy(self.samples[idx]).to(self.device)
		C = torch.from_numpy(self.samples_C[idx]).to(self.device)
		if self.normalize:
			X /= self.norm_x
			C /= self.norm_c
		return X, C

	@property
	def shape(self):
		return self.samples.shape

	@property
	def coefs_shape(self):
		return self.samples_C[0][0].shape

	@property
	def input_shape(self):
		return self.samples[0][0].shape