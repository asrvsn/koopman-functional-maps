import torch
import torch.optim as optim
from torch.autograd import Variable
import pywt
import matplotlib.pyplot as plt

from datasets import *
from preprocess import *
from functions import *
from utils import *
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# X = sinusoid(10000, 20)
X = duffing(10000, 3, 50)

print('==> Preparing data..')
coefs_fn = lambda X_t: coefs_wavelet(X_t, pywt.Wavelet('haar'))
inverse_fn = lambda C_t: inverse_wavelet(C_t, pywt.Wavelet('haar'))

train_set = TimeSeriesDataset(X, coefs_fn, device=device, normalize=2.0, n_samples=100, offset=5)
test_set = TimeSeriesDataset(X, coefs_fn, device=device, normalize=2.0, n_samples=10, offset=5)

c_shape = train_set.coefs_shape
x_shape = train_set.input_shape
n = len(train_set)
lr = 0.01
mom = 0.9

def train(epoch, model, optimizer):
	print('\nEpoch: %d' % epoch)
	optimizer.zero_grad()
	loss = 0
	for (X, C) in train_set:
		C0, C1 = C[0], C[1]
		X0, X1 = X[0], X[1]
		loss += model.loss(X0, X1)
	loss /= n
	loss.backward()
	optimizer.step()
	return loss.item()

def vis(model, i, dataset):
	with torch.no_grad():
		(X, C) = dataset[i]
		[X0, X1] = X
		X_pred = model.pred(X0)
		result = {
			'input': X0.cpu().numpy(),
			'target': X1.cpu().numpy(),
			'prediction': X_pred.cpu().numpy(),
		}
		return plot_images(result)

def test(model):
	with torch.no_grad():
		loss = 0
		for (X, C) in test_set:
			X0, X1 = X[0], X[1]
			C0, C1 = C[0], C[1]
			loss += model.loss(X0, X1)
		loss /= len(test_set)
		print(f'Test loss: {loss}')
		return loss

def run(early_stop=True):
	print('==> Learning..')
	model = Model3(x_shape, device)

	optimizer = optim.Adam(model.params, lr=lr)

	epoch = 0
	prev_grad = 0
	prev_loss = float('inf')
	loss_history = []

	try: 
		while True:
			loss = train(epoch, model, optimizer)
			print(f'Loss: {loss}')
			if early_stop and np.abs(prev_loss - loss) < 1e-3:
				break
			loss_history.append(loss)
			prev_loss = loss
			epoch += 1
	except KeyboardInterrupt:
		print('Quitting training early.')

	print('==> Testing..')
	test_loss = test(model)

	f1 = vis(model, random.randint(0, len(test_set)-1), test_set)
	f2 = plt.figure()
	plt.plot(np.arange(len(loss_history)), loss_history)
	# f3 = plt.figure()
	# plt.imshow(model.W.detach().cpu().numpy(), cmap=plt.cm.gray)

	plt.show()

if __name__ == '__main__':
	run(early_stop=False)
