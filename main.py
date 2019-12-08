import torch
import pywt

from functions import *
from preprocess import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
X = None # TODO
coefs_fn = lambda X_t: coefs_fourier(X_t)
inverse_fn = lambda C_t: inverse_fourier(C_t)
dataset = TimeSeriesDataset(X, coefs_fn, device=device)

print('==> Learning..')
c_shape = dataset.coefs_shape
n = len(dataset)
gamma = 0.01
ker = lambda X, Y: gaussian_kernel(X, Y)
ker_dx = lambda X, Y: gaussian_kernel(X, Y, der='x')
ker_dy = lambda X, Y: gaussian_kernel(X, Y, der='y')

def train(epoch, K):
    print('\nEpoch: %d' % epoch)
    grad = 0
    loss = 0
	for idx, (_, C) in dataset:
		[C0, C1] = C
		e0 = torch.matmul(K, C0)
		e11 = ker_dy(e0, e0)
		e12 = ker_dx(e0, e0)
		e13 = -2*ker_dx(e0, C1)
		grad += torch.matmul(C0.t(), e11+e12+e13)
		loss += torch.trace(ker(e0, e0) - 2*torch.matmul(K, ker(e0, C1)) + ker(C1, C1))
	grad /= n
	K -= gamma * grad
	loss /= n
    print('\nLoss: %d' % loss)
	return K

def pred(K, i):
	(X, C) = dataset[i]
	[X0, X1] = X
	[C0, _] = C
	C1 = torch.matmul(K, C0).cpu().numpy()
	X_pred = inverse_fn(C1)
	result = {
		'input': X0,
		'prediction': X_pred,
	}
	plot_images(result)

def run():
	epochs = 20
	K = torch.randn(c_shape[0], c_shape[0])
	for epoch in range(epochs):
		K = train(epoch, K)

	i = 10
	pred(K, i)