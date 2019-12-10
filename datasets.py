import numpy as np

from duffing import get_position_velocity
from preprocess import delay_embed

def sinusoid(n: int, d: int):
	'''
	1-dimensional sin() dataset
	'''
	x = np.linspace(-n/10, n/10, n)
	y = np.sin(2*x)
	X = delay_embed(y, 1, d)
	return X

def duffing(n: int, t: int, d: int):
	'''
	Forced Duffing oscillator.
	'''
	p, v = get_position_velocity(n)
	X = delay_embed(p, t, d)
	return X