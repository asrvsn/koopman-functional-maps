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

# X = sinusoid(10000, 20)
X = duffing(10000, 5, 50)

print('==> Preparing data..')
coefs_fn = lambda X_t: coefs_wavelet(X_t, pywt.Wavelet('haar'))
inverse_fn = lambda C_t: inverse_wavelet(C_t, pywt.Wavelet('haar'))
train_set = TimeSeriesDataset(X, coefs_fn, device=device, normalize=True, n_samples=100)
test_set = TimeSeriesDataset(X, coefs_fn, device=device, normalize=True, n_samples=10)

c_shape = train_set.coefs_shape
x_shape = train_set.input_shape
n = len(train_set)
lr = 0.01
mom = 0.9

gaussian_kernel = RFFKernel(x_shape[1], D=200, metric='rbf', device=device)
laplace_kernel = RFFKernel(x_shape[1], D=50, metric='laplace', device=device)

# fmap = gaussian_kernel
fmap = lambda x: x

def loss_fn(K, C0, C1):
	return torch.pow(torch.norm(fmap(torch.matmul(K, C0)) - fmap(C1), p='fro'), 2)
	# KC0 = torch.matmul(K, C0)
	# return torch.trace(ker(KC0, KC0)) - torch.trace(2*ker(KC0, C1)) + torch.trace(ker(C1, C1))

def train(epoch, K, prev_grad):
	print('\nEpoch: %d' % epoch)
	grad = 0
	loss = 0
	for (_, C) in train_set:
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
	for (X, C) in train_set:
		C0, C1 = C[0], C[1]
		X0, X1 = X[0], X[1]
		loss += loss_fn(K, C0, C1)
	loss /= n
	loss.backward()
	optimizer.step()
	return loss.item()

def pred(K, i, dataset):
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

def test(K):
	with torch.no_grad():
		loss = 0
		for (X, C) in test_set:
			X0, X1 = X[0], X[1]
			C0, C1 = C[0], C[1]
			KC0 = torch.matmul(K, C0)
			KX0 = inverse_fn(KC0.cpu().numpy())
			KX0 = torch.from_numpy(KX0).to(device)
			loss += torch.norm(X1 - KX0, p='fro').item()
		loss /= len(test_set)
		print(f'Test loss: {loss}')
		return loss

def vis(X, K, n=1000):
	fig = plt.figure()
	for i in range(n):
		X_t = X[i:i+X.shape[1]]
		C_t = coefs_fn(X_t)
		KC = np.dot(K, C_t)
		X_pred = inverse_fn(KC)
		

def run(early_stop=True):
	print('==> Learning..')
	K = torch.randn(c_shape[0], c_shape[0]).double().to(device)
	K = Variable(K, requires_grad=True)

	optimizer = optim.SGD([K], lr=lr, momentum=mom)

	epoch = 0
	prev_grad = 0
	prev_loss = float('inf')
	loss_history = []

	try: 
		while True:
			loss = train2(epoch, K, optimizer)
			print(f'Loss: {loss}')
			if early_stop and np.abs(prev_loss - loss) < 1e-3:
				break
			loss_history.append(loss)
			prev_loss = loss
			epoch += 1
	except KeyboardInterrupt:
		print('Quitting training early.')

	print('==> Testing..')
	test_loss = test(K)

	f1 = pred(K, random.randint(0, len(test_set)-1), test_set)
	f2 = plt.figure()
	plt.plot(np.arange(len(loss_history)), loss_history)
	f3 = plt.figure()
	K = K.detach().cpu().numpy()
	plt.imshow(K, cmap=plt.cm.gray)

	vis(X)

	plt.show()

if __name__ == '__main__':
	run(early_stop=False)
