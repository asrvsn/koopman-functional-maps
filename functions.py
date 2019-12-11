from scipy.stats import cauchy, laplace
import torch
import numpy as np

def acos_safe(x, eps=1e-4):
    return x.clamp(-1.0 + eps, 1.0 - eps).acos()

class RFFKernel:
	'''
	Random Fourier features implementation of Gaussian/Laplacian kernel.
	transcribed from https://github.com/hichamjanati/srf
	'''
	def __init__(self, d, gamma=1, D=50, metric='rbf', device=torch.device('cpu')):
		'''
		d: dimension of input
		D: number of Fourier features
		metric: 'rbf' or 'laplace'
		'''
		self.D = D

		# Sample frequencies
		if metric == 'rbf':
			self.w = np.sqrt(2*gamma)*torch.empty((D, d), device=device).normal_().double()
		elif metric == 'laplace':
			self.w = cauchy.rvs(scale=gamma, size=(D,d))
			self.w = torch.from_numpy(self.w).double().to(device)
		self.w_inv = torch.pinverse(self.w)

		# Sample offsets
		self.b = 2*np.pi*torch.empty((D,), device=device, dtype=torch.double).random_(0, to=1)

	def __call__(self, X, inv=False):
		if inv:
			return torch.matmul(acos_safe(X/(np.sqrt(2/self.D))) - self.b.unsqueeze(0), self.w_inv.t())
		else:
			return np.sqrt(2/self.D)*torch.cos(torch.matmul(X, self.w.t()) + self.b.unsqueeze(0))

def gaussian_kernel(X, Y, sigma=None, der=None):
	'''
	X and Y must be same dimensions.
	'''
	if sigma is None:
		sigma = 1.0 / X.shape[0]

	fn = lambda x, y: torch.exp(-torch.pow(torch.norm(x-y, p=2), 2) / (sigma ** 2))
	u = inner_product(X, Y, fn)

	if der == 'x' or der == 'y':
		du = -2*(X-Y)/(sigma ** 2)
		return u * du
	else:
		return u 

def poly_kernel(X, Y, degree=3, gamma=None, coef0=1, der=None):
	'''
	X and Y must be same dimensions.
	'''
	if gamma is None:
		gamma = 1.0 / X.shape[0]
	if der == 'x':
		return degree*gamma*Y*torch.pow(gamma*torch.matmul(X.t(), Y) + coef0, degree-1)
	elif der == 'y':
		return degree*gamma*X*torch.pow(gamma*torch.matmul(X.t(), Y) + coef0, degree-1)
	else:
		return torch.pow(gamma * torch.matmul(X.t(), Y) + coef0, degree)

def linear_kernel(X, Y, der=None):
	'''
	x and y must be same dimensions.
	'''
	if der == 'x':
		return Y
	elif der == 'y':
		return X
	else:
		return torch.matmul(X.t(), Y)

def inner_product(X, Y, fn):
	'''
	Custom inner product for two matrices X, Y. TODO: horribly inefficient..
	'''	
	n = X.shape[1]
	out = torch.zeros((n, n))
	for i in range(n):
		for j in range(n):
			out[i,j] = fn(X[:,i], Y[j,:])
	return out