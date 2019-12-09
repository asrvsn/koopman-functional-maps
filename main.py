import torch
import torch.optim as optim
from torch.autograd import Variable
import pywt
import matplotlib.pyplot as plt

from datasets import *
from preprocess import *
from functions import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = sinusoid_dataset(10000, 20)

print('==> Preparing data..')
coefs_fn = lambda X_t: coefs_wavelet(X_t, pywt.Wavelet('haar'))
inverse_fn = lambda C_t: inverse_wavelet(C_t, pywt.Wavelet('haar'))
dataset = TimeSeriesDataset(X, coefs_fn, device=device, normalize=True)

print('==> Learning..')
c_shape = dataset.coefs_shape
n = len(dataset)
lr = 0.01
mom = 0.9
ker = lambda X, Y: linear_kernel(X, Y)
ker_dx = lambda X, Y: linear_kernel(X, Y, der='x')
ker_dy = lambda X, Y: linear_kernel(X, Y, der='y')

def train(epoch, K, prev_grad):
	print('\nEpoch: %d' % epoch)
	grad = 0
	loss = 0
	for (_, C) in dataset:
		C0, C1 = C[0], C[1]
		e0 = torch.matmul(K, C0)
		e11 = ker_dy(e0, e0)
		e12 = ker_dx(e0, e0)
		e13 = -2*ker_dx(e0, C1)
		grad += torch.matmul(e11+e12+e13, C0.t())
		loss += torch.pow(torch.norm(torch.matmul(K, C0) - C1, p='fro'), 2)
		# loss += torch.trace(ker(e0, e0)) - 2*torch.trace(ker(e0, C1)) + torch.trace(ker(C1, C1))
	grad /= n
	K -= mom * prev_grad + lr * grad
	loss /= n
	loss = loss.item()
	print('\nLoss: %d' % loss)
	return K, loss, grad

def train2(epoch, K, optimizer):
	print('\nEpoch: %d' % epoch)
	optimizer.zero_grad()
	loss = 0
	for (_, C) in dataset:
		C0, C1 = C[0], C[1]
		loss += torch.pow(torch.norm(torch.matmul(K, C0) - C1, p='fro'), 2)
	loss /= n
	loss.backward()
	optimizer.step()
	return loss.item()

def pred(K, i):
	with torch.no_grad():
		(X, C) = dataset[i]
		[X0, X1] = X
		[C0, _] = C
		C1 = torch.matmul(K, C0).cpu().numpy()
		X_pred = inverse_fn(C1)
		result = {
			'input': X0.cpu().numpy(),
			'target': X1.cpu().numpy(),
			'prediction': X_pred,
		}
		return plot_images(result)

def run():
	epochs = 100
	K = torch.randn(c_shape[0], c_shape[0]).double().to(device)
	K = Variable(K, requires_grad=True)

	optimizer = optim.SGD([K], lr=lr, momentum=mom)

	prev_grad = 0
	prev_loss = float('inf')
	loss_history = []
	for epoch in range(epochs):
		loss = train2(epoch, K, optimizer)
		print('\nLoss: %d' % loss)
		loss_history.append(loss)
		if np.abs(prev_loss - loss) < 1e-3:
			break
		prev_loss = loss

	i = 10
	f1 = pred(K, i)
	f2 = plt.figure(2)
	plt.plot(np.arange(len(loss_history)), loss_history)
	f3 = plt.figure(3)
	K = K.detach().cpu().numpy()
	plt.imshow(K, cmap=plt.cm.gray)
	plt.show()

if __name__ == '__main__':
	run()