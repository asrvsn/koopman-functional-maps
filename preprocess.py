import numpy as np
import random
import pywt
from torch.utils.data import Dataset, DataLoader

def block_bootstrap_example(X: np.ndarray, block_size: int):
	'''
	Samples X_t by block bootstrap.
	X: numpy tensor (m channels x n time steps)
	block_size: size of each block
	'''
	if len(X.shape) > 2:
		# TODO
		raise Exception('Cant process more than 1D samples').

	t = random.randint(0, X.shape[1]-block_size-1)
	X1, X2 = X[t:t+block_size], X[t+1:t+block_size+1]
	return X1, X2

def coefs_fourier(X_t: np.ndarray):
	'''
	Compute fourier coefficients for sample.
	X_t: sample at a particular time
	d: number of kernel parameters to try 
	'''
	return np.fft.fftn(X_t)

def inverse_fourier(C_t: np.ndarray):
	return np.fft.ifftn(C_t)

def coefs_wavelet(X_t: np.ndarray, wavelet: pywt.Wavelet):
	'''
	Compute wavelet coefficients for sample.
	X_t: sample at a particular time
	wavelet: pywt.Wavelet()
	'''
	if len(X_t.shape) > 2:
		# TODO
		raise Exception('Cant process more than 1D samples').
	cA, (cH, cV, cD) = pywt.dwt2(X_t, wavelet, mode='smooth')
	return np.block([[cA, cH], [cV, cD]])

def inverse_wavelet(C_t: np.ndarray, wavelet: pywt.Wavelet):
	(m, n) = C_t.shape
	h, v = int(m/2), int(n/2)
	cA, cH, cV, cD = C_t[:h, :v], C_t[:h, v:], C_t[h:, :v], C_t[h:, v:]		
	return pywt.idwt2((cA, (cH, cV, cD)), wavelet, mode='smooth')

class TimeSeriesDataset(Dataset):
	def __init__(self, X: np.ndarray, coefs_fn, block_size=None, n_samples=None, device=torch.device("cpu")):
		'''
		Data loader for time series.
		'''
		m, n = X.shape[0], X.shape[1] # TODO multi-dimensional samples
		if block_size is None:
			block_size = m
		if n_samples is None:
			n_samples = int(n / block_size)

		self.X = X
		self.n = n_samples
		self.device = device

		self.samples = np.arange(n_samples)
		self.samples_C = np.arange(n_samples)
		for i in range(n_samples):
			X1, X2 = block_bootstrap_example(X, block_size)
			C1, C2 = coefs_fn(X1), coefs_fn(X2)
			self.samples[i] = [X1, X2]
			self.samples_C[i] = [C1, C2]

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		X = torch.from_numpy(self.samples[idx]).to(self.device)
		C = torch.from_numpy(self.samples_C[idx]).to(self.device)
		return X, C

	@property
	def coefs_shape(self):
		return self.samples_C[0].shape
