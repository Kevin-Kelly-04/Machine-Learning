import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random, math

def generateData(classA, classB):
	inputs = np.concatenate((classA , classB))
	targets = np.concatenate(
		(np.ones(classA.shape[0]),
		-np.ones(classB.shape[0])))

	N = inputs.shape[0] # Number of rows ( s a m p l e s )

	permute=list(range(N))
	random.shuffle(permute)
	inputs = inputs[permute, :]
	targets = targets[permute]

	return inputs, targets, N

def linearKernel(x, y):
	return np.dot(x, y)

def polynomialKernel(x, y):
	p = 2
	x_T = np.transpose(x)
	temp = np.dot(x_T, y) + 1
	return(np.power(temp, p))

def radialKernel(x, y):
	sigma = 9
	xy_norm = np.linalg.norm(np.subtract(x, y))
	xy_norm = np.power(xy_norm, 2)
	smooth_xy = xy_norm/(2*(np.power(sigma, 2)))
	return math.exp(-smooth_xy)

def objective(alpha):
	temp = 0.5*(np.dot(alpha, np.dot(alpha, p_matrix)))
	return temp - np.sum(alpha)

def zerofun(alpha):
	return(np.dot(alpha, targets))

def biasCalc(non_zero):
	bias = 0
	for elem in non_zero:
		bias += elem[0] * elem[2] * kernel(elem[1], non_zero[0][1])
	return (bias - non_zero[0][2])

def indicator(x, y, non_zero):
	bias = biasCalc(non_zero)
	ind = 0
	for elem in non_zero:
		ind += elem[0] * elem[2] * kernel([x, y], elem[1])
	return (ind - bias)

# generate class data 
np.random.seed(100)
classA = np.concatenate(
		(np.random.randn(20, 2) * 0.30 + [1.5, 0.0],
		np.random.randn(10, 2) * 0.20 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.20 + [0.0, -0.5]

# get inputs and target
inputs = np.concatenate((classA , classB))
targets = np.concatenate(
	(np.ones(classA.shape[0]),
	-np.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows ( s a m p l e s )

permute=list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

p_matrix = np.zeros((N, N))
kernel = radialKernel

for i in range(N):
	for j in range(N):
		p_matrix[i][j] = targets[i]*targets[j]*kernel(inputs[i], inputs[j])

#objective = objective(alpha, p_matrix)

C = 15
start = np.zeros(N)
XC = {'type':'eq', 'fun':zerofun}
#B = [(0, None) for b in range(N)]
B = [(0, C) for b in range(N)]
ret = minimize(objective, start, bounds=B, constraints=XC)

if (not ret['success']): # The string 'success' instead holds a boolean representing if the optimizer has found a solution
    raise ValueError('Cannot find optimizing solution')
else:
	print("solution")

alpha = ret['x']
non_zero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if alpha[i] > 10e-5]

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal') # force same scale on both axes
#plt.plot([p[1][0] for p in non_zero], [p[1][1] for p in non_zero], 'yo')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(x, y, non_zero) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
#plt.savefig('svmplot.pdf') # save copy in a file
plt.show() #show plot on screen

