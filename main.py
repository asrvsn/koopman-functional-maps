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
lag = 1
embed_dim = 10
# X = duffing(100, lag, embed_dim)
X = fitzhugh_nagumo(1000, lag, embed_dim)

print('==> Preparing data..')
coefs_fn = lambda X_t: coefs_wavelet(X_t, pywt.Wavelet('haar'))
inverse_fn = lambda C_t: inverse_wavelet(C_t, pywt.Wavelet('haar'))

train_set = TimeSeriesDataset(X, coefs_fn, device=device, normalize=2.0, n_samples=100, offset=3)
test_set = TimeSeriesDataset(X, coefs_fn, device=device, normalize=2.0, n_samples=10, offset=3)

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

def vis_data(X):
	x1, x2 = un_embed_2d(X, lag) # TODO nd
	plt.figure()
	plt.title('Sample')
	plt.plot(x1, x2)

def extrapolate(model, extent):
	with torch.no_grad():
		skip = int(len(X) / extent)
		x = X[0]
		shape = (extent, x.shape[0], x.shape[1])
		trajectory = np.empty(shape)
		trajectory[0] = x
		for i in range(1, extent):
			x = torch.from_numpy(X[i*skip]).unsqueeze(1).to(device)
			x = model.pred(x)
			trajectory[i] = x.squeeze(1).cpu().numpy()
		x1, x2 = un_embed_2d(trajectory, lag)
		plt.figure()
		plt.title('Predicted horizon')
		plt.plot(x1, x2, color='red')

def run(early_stop=True):
	print('==> Learning..')
	print(test_set.shape)
	model = Model1(x_shape, device)

	optimizer = optim.SGD(model.params, lr=lr, momentum=mom)

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

	# f1 = vis(model, random.randint(0, len(test_set)-1), test_set)
	f2 = plt.figure()
	plt.title(f'Convergence: {epoch} epochs')
	plt.yscale('log')
	plt.plot(np.arange(len(loss_history)), loss_history)
	# f3 = plt.figure()
	# plt.imshow(model.W.detach().cpu().numpy(), cmap=plt.cm.gray)

	vis_data(X)

	extrapolate(model, 1000)

	plt.show()

if __name__ == '__main__':
	run(early_stop=False)
