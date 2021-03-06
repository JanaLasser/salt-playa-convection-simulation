#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Two-dimensional simulation code for buoyancy driven convection below an 
evaporating salt lake. Allows for input of different simulation parameters
(see main.py -h for all available options) as well as different boundary
conditions.

This source code is subject to the terms of the MIT license. If a copy of the
MIT license was not distributed with this file, you can obtain one at
https://opensource.org/licenses/MIT

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
"""

__author__ = 'Jana Lasser'
__copyright__ = 'Copyright 2020, Geophysical pattern formation in salt playa'
__credits__ = ['Jana Lasser', 'Marcel Ernst']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Jana Lasser'
__email__ = 'lasser@csh.ac.at'
__status__ = 'Dev'


import os
from os.path import join
from os import getcwd
import sys
import argparse
from time import time

import numpy as np

from derivative import *
from printfunctions import *
from initialconditions import *

# Limit number of OMP threads used by numpy/scipy via OpenBLAS
os.environ['OMP_NUM_THREADS'] = '{:d}'.format(1)

# parameter-I/O
parser = argparse.ArgumentParser(description='Simulation of two-dimensional '\
    + 'porous media flow' +\
    ' \nCopyright (C) 2017 Marcel Ernst & Jana Lasser')

parser.add_argument('-dest', type=str, help='Complete path to the folder '\
                + 'results will be saved to if different than script location',
                default=getcwd())

parser.add_argument('-Ra','--rayleigh', type=float, help='Rayleigh number of'+\
				' the system', default=100)

parser.add_argument('-Ra2','--rayleigh2', type=int, help='Second Rayleigh number of'+\
				' the system', default=100)

parser.add_argument('-H','--height', type=int, help='Height of the '\
                + 'system in natural units',\
                default=10)

parser.add_argument('-L','--length', type=int, help='Length of the '\
                + 'system in natural units',\
                default=10)

parser.add_argument('-res','--resolution',type=int,\
                    help='number of grid cells per unit of length.',\
                    default=6)

parser.add_argument('-T','--maximum_simulation_time',type=float, \
					help='duration of simulation in natural units',\
					default=25.0)

parser.add_argument('-savetime','--saving_time_interval',type=float,\
                    help='Interval in which simulation output is dumped',\
                    default=0.1)

parser.add_argument('-clf', '--adaptive_time_constant', type=float,\
					help='CLF-constant for adaptive timestepping',\
					default = 0.05)

parser.add_argument('-plt','--plot_field',action="store_true",\
                help='Plots fields for debugging / movie creation',default=False)

parser.add_argument('-saveall','--save_all_fields',action="store_true",\
                help='Saves velocity fields in addition to salinity',default=False)

parser.add_argument('-amplitude','--wave_amplitude',type=float, \
                help='Amplitude of sinusoidal variations in evaporation' + \
                ' rate at the surface boundary condition',default=0.0)

parser.add_argument('-waves','--wave_number',type=int,\
                help='Number of sinusoidal variations in evaporation' + \
                ' rate at the surface boundary condition',default=1)

parser.add_argument('-init','--initial_condition',type=str,\
                help='Type of initial conditions. Can be eather "std" for' + \
                ' the steady state condition or "dyn" for an exponential decay' +\
                ' with a length scale from fits to the salinity of the convecting system',\
                default='std')

parser.add_argument('-S','--seed',type=int,\
                help='Seed for the random number generator. Per default, '+\
                'a random seed will be generated based on system time',default=-1)


args = parser.parse_args()

# System control parameter
RA = args.rayleigh # Rayleigh number
RA2 = args.rayleigh2 # second Rayleigh number in case we want a split system
if RA / int(RA) == 1:
	RA = int(RA)

# seed for random number generator. Will be randomly generated based
# on system time per default. Use custom seed to re-run simulations
# and investigate simulation crashes
seed = args.seed
if seed == -1:
	seed = int(time())

# Space constants
HEIGHT = args.height # height in natural units
LENGTH = args.length # length in natural units
#A = args.aspect_ratio # aspect ratio of length to HEIGHT = L/H
#LENGTH = HEIGHT*A
A = float(LENGTH/HEIGHT)
res = args.resolution # number of grid cells / unit of length

# Time constants
MAXTIME = args.maximum_simulation_time # maximum simulation time in natural units
SAVETIME = args.saving_time_interval                                                    
adaptive_dt_constant = args.adaptive_time_constant
SAVECOUNTER = 0
dt = 0.0001 # initial value for dt, will be adapted later
global_time = 0.0 # initial value for global time

# Upper boundary parameters: 
# amplitude = amplitude of sinusoidal evaporation rate E(X)
# waves = number of waves of E(X) in the box
# phi = initial phase shift of these waves and the convection cell
amplitude = args.wave_amplitude
waves = args.wave_number

initial_condition = args.initial_condition

parameters = {'Ra': RA, 'Ra2':RA2, 'A': A	 , \
	'amplitude': amplitude, 'waves': waves, 'phi': 0.0, \
	'max_T':MAXTIME, 'clf':adaptive_dt_constant, 'res':res,\
	'HEIGHT':HEIGHT, 'LENGTH':LENGTH, 'initial conditiions':initial_condition}

# I/O handling
dest = args.dest # location of results folder
run = 1
run_name = 'Ra{}_{}x{}_res{}_T{}_clf{}_amp{}_waves{}_run{}'.format(RA, HEIGHT, \
	int(LENGTH), res, MAXTIME, adaptive_dt_constant,amplitude, waves, run)
SAVEPATH = join(dest, run_name)
plot_field = args.plot_field
save_all_fields = args.save_all_fields

# size of box (natural units) 
length = np.array([HEIGHT*parameters['A'],HEIGHT]) #2d

# number of grid points
SIZE_Y =  HEIGHT *  res
size = np.array([int(SIZE_Y*parameters['A']), SIZE_Y]) #2d
size_b = np.array([size[0], size[1]-2]) #size without top/bottom boundaries

# grid spacing
dx = np.divide(length, np.array([size[0], size[1]-1])) #2d

# Create Rayleigh matrix for height dependent Rayleigh number
Ra = Rayleigh_Matrix(size, length, parameters)

# create savepath which will serve as root directory for
# the simulation results. Results for the three fields will
# be saved to three subfolders 'C', 'Ux' and 'Uz'
while os.path.exists(SAVEPATH):
	run += 1
	run_name = 'Ra{}_{}x{}_res{}_T{}_clf{}_amp{}_waves{}_run{}'.format(RA, HEIGHT, \
	int(LENGTH), res, MAXTIME, adaptive_dt_constant,amplitude, waves, run)
	SAVEPATH = join(dest, run_name)

# create directories for data storage
print('\n\nwriting to {}'.format(SAVEPATH))
fields = ['C','Ux','Uz']
os.makedirs(SAVEPATH)
for field in fields:
	os.makedirs(join(SAVEPATH,field))

# values of boundaries for concentration (C from 0 to 1)
boundaries_C = [1., 0.] 

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

### helper functions

# derivative in x- and y- direction
def Derivative(F, direction):
	if direction in ['dx', 'dxx']:
		return np.matmul(derivatives[direction], F)
	elif direction in ['dy', 'dyy']:
		return np.transpose(np.matmul(derivatives[direction], np.transpose(F)))

# derivative in y-direction for the concentration applying boundaries_C
def Dirichlet_Derivative_C(F, direction): 
	b = np.zeros((size[1]-2,))
	for i in range(2):
		b += dirichlet_vectors[direction][i]*boundaries_C[i]
	return np.transpose(np.matmul(derivatives[direction], np.transpose(F))) + b


# define the initial conditions
def InitialConditions(size, par, dx):
	if initial_condition == 'std':
		# load Steady-State
		C = Load_SteadyStateC(size, dx, length)
	elif initial_condition == 'dyn':
		C = Load_DynamicDecayC(size, dx, length, parameters['Ra'])
	elif initial_condition not in ('std', 'dyn') and parameters['Ra'] == 100:
		decay_length_scale = float(initial_condition)
		C = Load_SpecificDecayC(size, dx, length, decay_length_scale)
	else:
		print('initial condition unknown, terminating...')
		sys.exit()

	# load from file
	#C = Load_InitialC('onecell200.npy', size= size, frequency = parameters['waves'])

	# add random noise
	C = AddRandomNoise(C, size, seed, factor = 0.05)
	
	# initialize velocity fields in x and z direction as zero
	Psi = np.zeros(size_b, dtype=np.float32)
	Omega = np.zeros(size_b, dtype=np.float32)
	Psik = np.zeros(size_b, dtype = np.complex)

	U = np.array([Derivative(Psi, 'dy') ,-1 - Derivative(Psi, 'dx')])
	return (U, C, Psi, Omega, Psik)

# solve advection-diffusion equation in real space
def RungeKutta(U, C, dx, dt):
	def f(U,C):
		C_dx = Derivative(C, 'dx')
		C_dy = Dirichlet_Derivative_C(C, 'dy')
		C_dxx = Derivative(C, 'dxx')
		C_dyy = Dirichlet_Derivative_C(C, 'dyy')
		return C_dxx + C_dyy - (U[0]*C_dx + U[1]*C_dy) # diffusion - advection

	# classical Runge-Kutta method 4th order
	k1 = f(U,C[:,1:-1])
	k2 = f(U,C[:,1:-1] + (dt / 2.0) * k1)
	k3 = f(U,C[:,1:-1] + (dt / 2.0) * k2)
	k4 = f(U,C[:,1:-1] + dt * k3)
	C[:,1:-1] += dt / 6.0 * (k1 + 2* k2 + 2 * k3 + k4)

	return C

# primary function for the time stepping
def IntegrationStep(U, C, Psi, Psik, Omega, par, dx, global_time, dt):
	# Compute Omega from Ra and concentration Matrix C in real space
	Omega = +Ra[:,1:-1] * Derivative(C[:,1:-1], 'dx') - Omega0

	# compute Omega in Fourier space
	Omegak = np.fft.fftshift(np.fft.fft(-Omega, axis = 0), axes = 0)

	# compute Psi in Fourier space
	for kx in range(size[0]):
		Psik[kx,:] = np.matmul(tensor_fft[:,:,kx],Omegak[kx,:])

	# compute Psi in real space
	Psi = np.real(np.fft.ifft(np.fft.ifftshift(Psik, axes = 0), axis = 0))

	# compute velocity in real space
	U = np.array([Derivative(Psi, 'dy') ,- Derivative(Psi, 'dx')]) + U0

	# solve advection-diffusion equation in real space
	C = RungeKutta(U, C, dx, dt)
	global_time += dt
	return (U, C, Psi, Omega, global_time)

	

# define initial conditions
print('random seed: {}'.format(seed))
parameters.update({'seed':seed})
(U, C, Psi, Omega, Psik) = InitialConditions(size, parameters, dx)
U0, Omega0 = SinusoidalEvaporation(size_b, length, parameters)

# print a file recording simulation parameters and random seed
# into the simulation result root directory
PrintParams(parameters, SAVEPATH, run_name)

adaptive_time_counter = 0
while global_time < MAXTIME:
	(U, C, Psi, Omega, global_time) = IntegrationStep(U, C,\
		 Psi, Psik, Omega, parameters, dx, global_time, dt)	

	#adapt the time step
	if (adaptive_time_counter % 20 ==0)  or (adaptive_time_counter < 10):
		dt = dx[1] / np.max(np.abs(U[1])) * adaptive_dt_constant

	adaptive_time_counter += 1
	
	# save system state every SAVETIME:
	# all three fields C, Ux, Uz will be saved in binary format 
	if global_time > SAVECOUNTER*SAVETIME:
		print('current time: {}, dt = {}'\
			.format(round(global_time,4), dt))
		PrintField(C, global_time, SAVETIME, 'C', savepath=join(SAVEPATH,'C'))
		if save_all_fields:
			PrintField(U[1,0:,0:], global_time, SAVETIME, 'Uz', savepath=join(SAVEPATH,'Uz'))
			PrintField(U[0,0:,0:], global_time, SAVETIME, 'Ux', savepath=join(SAVEPATH,'Ux'))

		# plot field for debugging reasons (publication quality plots
		# are created separately from the field data)
		if plot_field:
			PlotField(C, global_time, SAVETIME, 'C', savepath=join(SAVEPATH,'C'))
			PlotField(U[1,0:,0:], global_time, SAVETIME, 'Uz', savepath=join(SAVEPATH,'Uz'))
			PlotField(U[0,0:,0:], global_time, SAVETIME, 'Ux', savepath=join(SAVEPATH,'Ux'))

		SAVECOUNTER += 1
