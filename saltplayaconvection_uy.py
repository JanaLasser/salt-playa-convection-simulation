## Limit number of OMP threads used by numpy/scipy via OpenBLAS
import os
os.environ['OMP_NUM_THREADS'] = '{:d}'.format(1)

import numpy as np
#from scipy.fftpack import fftn, ifftn, dst, idst, fft, ifft

#from scipy.linalg import solve_sylvester
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from derivative import CreateCFDMatrix#, CreateFFTMatrix
from np_derivatives import CreateNP_CFDMatrix, Create_FFTMatrix
from printfunctions import *
from initialconditions import *
#from matplotlib.backends.backend_pdf import PdfPages
import sys
###################

# Constants for time stepping (natural units)
MAXTIME = 25.
SAVETIME = 0.1
SAVECOUNTER = 0
dt = 0.0001
# CFL constant
adaptive_dt_constant = 0.05 # if it is not stable, reduce this number
global_time = 0.0


#system parameters: Rayleigh = Ra, lower layer Rayleigh  = Ra + Ra2
# A = aspect ratio = L/H
# amplitude = amplitude of sinusoidal evaporation rate E(X)
# waves = number of waves of E(X) in the box
# phi = initial phase shift of these waves and the convection cell
parameters = {'Ra': int(sys.argv[2]), 'Ra2' : 0., 'A': 1.0	 , \
	'amplitude': 0., 'waves': 1., 'phi': 0.0}

# size of box (natural units)
HEIGHT = 40 #height in natural units
length = np.array([HEIGHT*parameters['A'],HEIGHT]) #2d
res = 8

# number of grid points
SIZE_Y =  HEIGHT *  res
size = np.array([int(SIZE_Y*parameters['A']), SIZE_Y]) #2d
size_b = np.array([size[0], size[1]-2]) #size without top/bottom boundaries

#grid spacing
dx = np.divide(length, np.array([size[0], size[1]-1])) #2d

# Create Rayleigh matrix for height dependent Rayleigh number
Ra = Rayleigh_Matrix(size, length, parameters)

# create savepath
SAVEPATH = sys.argv[1] + '/'
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

#values of boundaries for concentration (C from 0 to 1)
boundaries_C = [1., 0.] #!!!!!

# 6th order coefficients
coeff_dx    = ([1., 1./3],       [14./9., 1./9.]     , 1, 'periodic')
# 6th order coefficients
coeff_dxx   = ([1. , 2./11],     [12./11, 3./11]     , 2, 'periodic') 

matrix_dy, dirichlet_vector_dy = CreateNP_CFDMatrix(axis = 1, \
	size = size, dx = dx, grade = 1, order = 6)
matrix_dyy, dirichlet_vector_dyy = CreateNP_CFDMatrix(axis = 1, \
	size = size, dx = dx, grade = 2, order = 6)
matrix_dx = CreateCFDMatrix(input = coeff_dx, \
	size = size, dx = dx, axis = 0)
matrix_dxx = CreateCFDMatrix(input = coeff_dxx, \
	size = size, dx = dx, axis = 0)
tensor_fft = Create_FFTMatrix(size = size, dx = dx)	

derivatives = {'dx' : matrix_dx, 'dxx': matrix_dxx, \
		'dy': matrix_dy, 'dyy': matrix_dyy}

dirichlet_vectors = {'dy': dirichlet_vector_dy, 'dyy': dirichlet_vector_dyy}

# Derivative in x- and y- direction
def Derivative(F, direction):
	if direction in ['dx', 'dxx']:
		return np.matmul(derivatives[direction], F)
	elif direction in ['dy', 'dyy']:
		return np.transpose(np.matmul(derivatives[direction], np.transpose(F)))

# Derivative in y-direction for the concentration applying boundaries_C
def Dirichlet_Derivative_C(F, direction): 
	b = np.zeros((size[1]-2,))
	for i in range(2):
		b += dirichlet_vectors[direction][i]*boundaries_C[i]
	return np.transpose(np.matmul(derivatives[direction], np.transpose(F))) + b



#### Define the initial conditions
def InitialConditions(size, par, dx):
	#Load Steady-State
	C = Load_SteadyStateC(size, dx, length)
	# Load from file
	#C = Load_InitialC('onecell200.npy', size= size, frequency = parameters['waves'])

	#Add Random noise
	C = AddRandomNoise(C, size, factor = 0.05)
	
	#initialize velocity fields in x and z direction as zero
	Psi = np.zeros(size_b, dtype=np.float32)
	Omega = np.zeros(size_b, dtype=np.float32)
	Psik = np.zeros(size_b, dtype = np.complex)

	U = np.array([Derivative(Psi, 'dy') ,-1 - Derivative(Psi, 'dx')])
	return (U, C, Psi, Omega, Psik)

# Solve Advection-Diffusion equation in real space
def RungeKutta(U, C, dx, dt):
	def f(U,C):
		C_dx = Derivative(C, 'dx')
		C_dy = Dirichlet_Derivative_C(C, 'dy')
		C_dxx = Derivative(C, 'dxx')
		C_dyy = Dirichlet_Derivative_C(C, 'dyy')
		return C_dxx + C_dyy - (U[0]*C_dx + U[1]*C_dy) # diffusion - advection

	## classical Runge–Kutta method, 4th order
	k1 = f(U,C[:,1:-1])
	k2 = f(U,C[:,1:-1] + (dt / 2.0) * k1)
	k3 = f(U,C[:,1:-1] + (dt / 2.0) * k2)
	k4 = f(U,C[:,1:-1] + dt * k3)
	C[:,1:-1] += dt / 6.0 * (k1 + 2* k2 + 2 * k3 + k4)

	## 3rd order Runge-Kutta (Wray 1991)
	# more stable!
	#k1 = C[:,1:-1] + dt*8./15*f(U,C[:,1:-1])
	#k2 = k1 + dt*(5./12*f(U,k1) - 17./60 * f(U,C[:,1:-1]))
	#C[:,1:-1] = k2 + dt*(3./4*f(U,k2) - 5./12 * f(U,k1))
	#print(global_time)
	return C


# primary function for the time stepping
def IntegrationStep(U, C, Psi, Psik, Omega, par, dx, global_time, dt):
	# Compute Omega from Ra and concentration Matrix C in real space
	Omega = +Ra[:,1:-1] * Derivative(C[:,1:-1], 'dx') - Omega0

	#Psi = solve_sylvester(derivatives['dxx'], derivatives['dyy'], -Omega)

	# Compute Omega in Fourier space
	Omegak = np.fft.fftshift(np.fft.fft(-Omega, axis = 0), axes = 0)
	# Compute Psi in Fourier space
	for kx in range(size[0]):
		Psik[kx,:] = np.matmul(tensor_fft[:,:,kx],Omegak[kx,:])
	# Compute Psi in real space
	Psi = np.real(np.fft.ifft(np.fft.ifftshift(Psik, axes = 0), axis = 0))

	# Compute velocity in real space
	U = np.array([Derivative(Psi, 'dy') ,- Derivative(Psi, 'dx')]) + U0

	# solve advection-diffusion equation in real space
	C = RungeKutta(U, C, dx, dt)
	global_time += dt
	return (U, C, Psi, Omega, global_time)


# Define initial conditions
(U, C, Psi, Omega, Psik) = InitialConditions(size, parameters, dx)
U0, Omega0 = SinusoidalEvaporation(size_b, length, parameters)
NumberOfMaxima = np.zeros((int(MAXTIME/SAVETIME)+1,size[1]), dtype = np.int)

adaptive_time_counter = 0
while global_time < MAXTIME:
	(U, C, Psi, Omega, global_time) = IntegrationStep(U, C,\
		 Psi, Psik, Omega, parameters, dx, global_time, dt)	

	#adapt the time step
	if (adaptive_time_counter % 20 ==0)  or (adaptive_time_counter < 10):
		dt = dx[1] / np.max(np.abs(U[1])) * adaptive_dt_constant
		#print(dt, adaptive_time_counter)
	adaptive_time_counter += 1
	
	# print stuff every SAVETIME
	if global_time > SAVECOUNTER*SAVETIME:
		print(round(global_time,2), dt, MAXTIME)
		PrintColorMatrix(Matrix = C, length = length, par = parameters, vmin = -0., vmax = 1.,\
			savepath = SAVEPATH, savename = ('C_' + str(float(length[0]))+'_'), time = global_time)
		#PrintCrossSection(Row = C[:,int(size[1]/2)].real, savepath = SAVEPATH, \
		#savename = ('C'+str(float(length[0]))+'_'),\
		#time = global_time, xlabel = 'x-direction', ylabel = 'concentration')

		savename='C_{}x{}_yres{:d}_Ra{:d}_dt{:1.4f}_T{}'.\
				format(HEIGHT, length, SIZE_Y, parameters['Ra'], dt, global_time)
		PrintConcentrationValues(C, parameters, savepath=SAVEPATH,\
			savename=savename)

		# Count number of maxima for each height and SAVETIME and save matrix

		NumberOfMaxima[SAVECOUNTER, :] = CountNumberOfMaxima(C)
		savename='NumberOfMaxima_{}x{}_yres{:d}_Ra{:d}'.\
				format(HEIGHT, length, SIZE_Y, parameters['Ra'])
		PrintConcentrationValues(NumberOfMaxima, parameters, savepath=SAVEPATH,\
			savename=savename, fmt = '%i')
		SAVECOUNTER += 1