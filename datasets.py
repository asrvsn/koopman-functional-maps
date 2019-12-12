import numpy as np

import duffing as duff
from neurons import *
from preprocess import *

def sinusoid(n: int, d: int):
	'''
	1-dimensional sin() dataset
	'''
	x = np.linspace(-n/10, n/10, n)
	y = np.sin(2*x)
	X = delay_embed(y, 1, d)
	return X

def duffing(n: int, lag: int, d: int):
	'''
	Forced Duffing oscillator.
	'''
	p, v = duff.get_position_velocity(n)
	X = np.array([p, v]).T
	X = delay_embed_2d(X, lag, d)
	return X

def fitzhugh_nagumo(n: int, lag: int, d: int):
	'''
	FN neuron model.
	'''
	# Parameters
	I_ampl = 0.324175
	V_0 = -0.6
	W_0 = -0.4

	# Create neuron and calculate trajectory
	ts, t_step = np.linspace(0, n, 100*n, retstep=True)
	neuron = FNNeuron(I_ampl=I_ampl, V_0=V_0, W_0=W_0)
	neuron.solve(ts=ts)

	X = np.array([neuron.Vs, neuron.Ws]).T
	X = delay_embed_2d(X, lag, d)
	return X

def speech(n: int, lag: int, d: int):
	'''
	Human speech dataset.
	'''
	# TODO
	pass