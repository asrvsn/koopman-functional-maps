import numpy as np
import torch
from torch.autograd import Variable

from functions import *

class Model1:
	def __init__(self, x_shape, device):
		W = torch.randn(x_shape).double().to(device)
		self.W = Variable(W, requires_grad=True)

	def pred(self, X):
		return torch.bmm(X, self.W)

	def loss(self, X0, X1):
		return torch.pow(torch.norm(self.pred(X0) - X1, p='fro'), 2)
		# KC0 = torch.matmul(K, C0)
		# return torch.trace(ker(KC0, KC0)) - torch.trace(2*ker(KC0, C1)) + torch.trace(ker(C1, C1))

	@property 
	def params(self):
		return [self.W]

class Model2:
	def __init__(self, x_shape, device):
		self.embed_dim = 100
		self.gk = RFFKernel(x_shape[1], D=self.embed_dim, metric='rbf', device=device)
		self.x_shape = x_shape
		self.device = device
		W = torch.randn(self.embed_dim, x_shape[1]).double().to(device)
		self.W = Variable(W, requires_grad=True)

	def pred(self, X):
		Z = self.gk(X)
		Y = acos_safe(Z/(np.sqrt(2/self.embed_dim))) - self.gk.b.unsqueeze(0)
		return torch.matmul(Y, self.W)

	def loss(self, X0, X1):
		return torch.pow(torch.norm(self.pred(X0) - X1, p='fro'), 2)
		# KC0 = torch.matmul(K, C0)
		# return torch.trace(ker(KC0, KC0)) - torch.trace(2*ker(KC0, C1)) + torch.trace(ker(C1, C1))

	@property 
	def params(self):
		return [self.W]

class Model3:
	def __init__(self, x_shape, device):
		self.embed_dim = 100
		self.x_shape = x_shape
		self.device = device

		b1 = 2*np.pi*torch.rand((self.embed_dim,), device=device, dtype=torch.double).unsqueeze(0)
		b2 = 2*np.pi*torch.rand((self.embed_dim,), device=device, dtype=torch.double).unsqueeze(0)
		W1 = torch.randn(x_shape[1], self.embed_dim).double().to(device)
		W2 = torch.randn(self.embed_dim, x_shape[1]).double().to(device)
		self.b1 = Variable(b1, requires_grad=True)
		self.b2 = Variable(b2, requires_grad=True)
		self.W1 = Variable(W1, requires_grad=True)
		self.W2 = Variable(W2, requires_grad=True)

	def pred(self, X):
		Z = np.sqrt(2/self.embed_dim) * torch.cos(torch.matmul(X, self.W1) + self.b1)
		Y = torch.matmul(acos_safe(Z/(np.sqrt(2/self.embed_dim))) - self.b2, self.W2)
		return Y

	def loss(self, X0, X1):
		return torch.pow(torch.norm(self.pred(X0) - X1, p='fro'), 2)
		# KC0 = torch.matmul(K, C0)
		# return torch.trace(ker(KC0, KC0)) - torch.trace(2*ker(KC0, C1)) + torch.trace(ker(C1, C1))

	@property 
	def params(self):
		return [self.b1, self.b2, self.W1, self.W2]

class Model4:
	def __init__(self, x_shape, device):
		self.embed_dim = 200
		self.x_shape = x_shape
		self.device = device

		b1 = 2*np.pi*torch.rand((self.embed_dim,), device=device, dtype=torch.double).unsqueeze(0)
		b2 = 2*np.pi*torch.rand((self.embed_dim,), device=device, dtype=torch.double).unsqueeze(0)
		W1 = torch.randn(x_shape[1], self.embed_dim).double().to(device)
		W2 = torch.randn(self.embed_dim, x_shape[1]).double().to(device)
		K =torch.randn(self.embed_dim, self.embed_dim).double().to(device)
		self.b1 = Variable(b1, requires_grad=True)
		self.b2 = Variable(b2, requires_grad=True)
		self.W1 = Variable(W1, requires_grad=True)
		self.W2 = Variable(W2, requires_grad=True)
		self.K = Variable(K, requires_grad=True)

	def pred(self, X):
		Z = np.sqrt(2/self.embed_dim) * torch.cos(torch.matmul(X, self.W1) + self.b1)
		Z = torch.matmul(Z, self.K)
		Y = torch.matmul(acos_safe(Z/(np.sqrt(2/self.embed_dim))) - self.b2, self.W2)
		return Y

	def loss(self, X0, X1):
		return torch.pow(torch.norm(self.pred(X0) - X1, p='fro'), 2)
		# KC0 = torch.matmul(K, C0)
		# return torch.trace(ker(KC0, KC0)) - torch.trace(2*ker(KC0, C1)) + torch.trace(ker(C1, C1))

	@property 
	def params(self):
		return [self.b1, self.b2, self.W1, self.W2, self.K]

# def train(epoch, K, prev_grad):
# 	print('\nEpoch: %d' % epoch)
# 	grad = 0
# 	loss = 0
# 	for (_, C) in train_set:
# 		C0, C1 = C[0], C[1]
# 		e0 = torch.matmul(K, C0)
# 		e11 = ker_dy(e0, e0)
# 		e12 = ker_dx(e0, e0)
# 		e13 = -2*ker_dx(e0, C1)
# 		grad += torch.matmul(e11+e12+e13, C0.t())
# 		loss += torch.pow(torch.norm(torch.matmul(K, C0) - C1, p='fro'), 2)
# 		# loss += torch.trace(ker(e0, e0)) - 2*torch.trace(ker(e0, C1)) + torch.trace(ker(C1, C1))
# 	grad /= n
# 	K -= mom * prev_grad + lr * grad
# 	loss /= n
# 	loss = loss.item()
# 	print('\nLoss: %d' % loss)
# 	return K, loss, grad
