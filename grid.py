#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def heatmap(a):
	"""a: 2D array to plot as a heatmap"""
	#plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.imshow(a, cmap='gray', interpolation='nearest')
	plt.show()


#  +-----------> x
#  | . . . . .
#  | . . . . .
#  | . . . . .
#  | . . . . .
#  | . . . . .
#  v
# y

gamma = 0.9

# States
w, h = 5, 5
grid = np.zeros((w, h))
n = grid.size
S = np.eye(n, n)

# Actions
k = 5
A = np.eye(k)
UP, DOWN, RIGHT, LEFT, NOOP = 0, 1, 2, 3, 4

# Transition probabilities
P = np.zeros((k, n, n))
# P[a, s2, s1] is the probability of ending in state s2 given that the agent
# starts in state s1 and chose action a.
# It is important to note that the initial state is the column index, not the
# row index, so that we can use vector representation of state, S2 = P[a] . S1
#
# P[UP, (x, y), (x, y + 1)] = 0.7
# P[UP, (x, y), (x, y - 1)] = 0.1
# P[UP, (x, y), (x + 1, y)] = 0.1
# P[UP, (x, y), (x - 1, y)] = 0.1
# P[UP, (x, y), (other)] = 0

# TODO: That transition tensor takes damn many lines to describe…
# Make it shorter, at least by somehow factorizing directions

# First build the random diffusion
grid.fill(1)
grid[-1,:] = 0
P[UP] += np.roll(S, shift=+w, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[:,0] = 0
P[UP] += np.roll(S, shift=-1, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[:,-1] = 0
P[UP] += np.roll(S, shift=+1, axis=0) * grid.reshape(-1)

# Normalize random diffusion and multiply it by 30%
P[UP] = normalize(P[UP], axis=0, norm='l1') * 0.3

# Add the main direction
grid.fill(1)
grid[0,:] = 0
P[UP] += np.roll(S, shift=-w, axis=0) * grid.reshape(-1) * 0.7

# Renormalize for cells in which random diffusion was 0
P[UP] = normalize(P[UP], axis=0, norm='l1')

# Idem for DOWN
grid.fill(1)
grid[0,:] = 0
P[DOWN] += np.roll(S, shift=-w, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[:,0] = 0
P[DOWN] += np.roll(S, shift=-1, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[:,-1] = 0
P[DOWN] += np.roll(S, shift=+1, axis=0) * grid.reshape(-1)
P[DOWN] = normalize(P[DOWN], axis=0, norm='l1') * 0.3
grid.fill(1)
grid[-1,:] = 0
P[DOWN] += np.roll(S, shift=+w, axis=0) * grid.reshape(-1) * 0.7
P[DOWN] = normalize(P[DOWN], axis=0, norm='l1')

# Idem for RIGHT
grid.fill(1)
grid[0,:] = 0
P[RIGHT] += np.roll(S, shift=-w, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[-1,:] = 0
P[RIGHT] += np.roll(S, shift=+w, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[:,0] = 0
P[RIGHT] += np.roll(S, shift=-1, axis=0) * grid.reshape(-1)
P[RIGHT] = normalize(P[RIGHT], axis=0, norm='l1') * 0.3
grid.fill(1)
grid[:,-1] = 0
P[RIGHT] += np.roll(S, shift=+1, axis=0) * grid.reshape(-1) * 0.7
P[RIGHT] = normalize(P[RIGHT], axis=0, norm='l1')

# Idem for LEFT
grid.fill(1)
grid[0,:] = 0
P[LEFT] += np.roll(S, shift=-w, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[-1,:] = 0
P[LEFT] += np.roll(S, shift=+w, axis=0) * grid.reshape(-1)
grid.fill(1)
grid[:,-1] = 0
P[LEFT] += np.roll(S, shift=+1, axis=0) * grid.reshape(-1)
P[LEFT] = normalize(P[LEFT], axis=0, norm='l1') * 0.3
grid.fill(1)
grid[:,0] = 0
P[LEFT] += np.roll(S, shift=-1, axis=0) * grid.reshape(-1) * 0.7
P[LEFT] = normalize(P[LEFT], axis=0, norm='l1')

P[NOOP] = np.eye(n)

# Example of test of P:
# > P[LEFT].dot(S[17]).reshape(w, h)
# This will show a 2D matrix representing the grid with the probability of
# being in the cells given that we were in state 17 and chosed action LEFT.

# "True" Reinforcement/Reward function, that we will try to guess.
grid.fill(0)
grid[0,-1] = 1
R = grid.reshape(-1)

# Observed agent action decision policy, from which we'll try to recover R
# (for the case w = h = 5)
policy = np.array([
	[RIGHT, RIGHT, RIGHT, RIGHT, NOOP],
	[UP,    RIGHT, RIGHT, UP,    UP  ],
	[UP   , UP   , UP   , UP   , UP  ],
	[UP   , UP   , RIGHT, UP   , UP  ],
	[UP   , RIGHT, RIGHT, RIGHT, UP  ],
]).reshape(-1)


# TODO: Linear pb to solve:
# max sum(i = 1..n, min(a = 1..k and a != policy[i], (P[policy[i]][i] - P[a][i]) . (I - gamma * P[policy[i]]).inverse() . R ) - lambd * norm1(R) )
# w/ (P[policy[i]][i] - P[a][i]) . (I - gamma * P[policy[i]]).inverse() . R >= 0 forall a != policy[i] 

