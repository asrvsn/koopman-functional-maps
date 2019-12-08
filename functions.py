import torch

def gaussian_kernel(x, y, sigma=None, der=None):
	if sigma is None:
		sigma = 1.0 / x.shape[0]
	u = torch.exp(-torch.pow(torch.norm(x-y, p=2), 2) / (sigma ** 2))
	if der == 'x' or der == 'y':
		du = -2*(x-y)/(sigma ** 2)
		return u * du
	else:
		return u 

def poly_kernel(x, y, degree=3, gamma=None, coef0=1, der=None):
	if gamma is None:
		gamma = 1.0 / x.shape[0]
	if der == 'x':
		return degree*gamma*y*torch.pow(gamma*torch.matmul(x, y) + coef0, degree-1)
	elif der == 'y':
		return degree*gamma*x*torch.pow(gamma*torch.matmul(x, y) + coef0, degree-1)
	else:
		return torch.pow(gamma * torch.matmul(x, y) + coef0, degree)

def linear_kernel(x, y, der=None):
	if der == 'x':
		return y
	elif der == 'y':
		return x
	else:
		return torch.matmul(x, y)
