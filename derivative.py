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

import numpy as np

### periodic derivative matrices ###

def CyclicDiagonalMatrix(entries, size):
	width = len(entries)
	M = np.zeros(shape=(size,size))
	for i in range(size):
		for d in range(width):
			M[i, (i+d-int(width/2))%size] = entries[d]
	return M

# input: derivation matrix coefficients
# size: tuple, correct size is choosen by axis kwarg
# dx: tuple for grid point distance in x- and y direction (natural units)
# in case of periodic boundaries in y-direction set axis = 1
def CreateCFDMatrix(input, size, dx, axis = 0): # Only periodic boundary conditions actually
	[left_coeff, right_coeff, order, boundary] = input[:]
	#left matrix
	size_axis = size[axis]
	N = len(left_coeff)
	left_entries = np.zeros(2*N-1)
	for i in range(2*N-1):
		left_entries[i] = left_coeff[abs(i-N+1)]

	#right matrix
	N = len(right_coeff)
	right_entries = np.zeros(2*N+1)
	if order == 1:
		for i in range(1,N+1):
			right_entries[N+i] =  right_coeff[i-1]*(1./(2*i*dx[axis]))
			right_entries[N-i] = -right_entries[N+i]
	elif order == 2:
		for i in range(1,N+1):
			right_entries[N+i] = right_coeff[i-1]*(1./(i*i*dx[axis]**2))
			right_entries[N-i] = right_entries[N+i]
			right_entries[N]  -= 2*right_entries[N+i]

	left_matrix = CyclicDiagonalMatrix(left_entries, size_axis)
	right_matrix = CyclicDiagonalMatrix(right_entries, size_axis)

	#Output derivative_matrix due to assoziativity
	inv_left_matrix = np.linalg.inv(left_matrix)
	derivative_matrix = np.matmul(inv_left_matrix,right_matrix)
	return derivative_matrix

### non-periodic derivative matrices ###

import numpy as np

# order: obsolete
# grade: 1st or 2nd diff
# inplements dirichlet boundary conditions (top/bottom)
def CreateNP_CFDMatrix(axis, size, dx, grade = 1, order = 6):
    s_ax = size[axis]
    LH_Matrix = np.zeros(shape=(s_ax, s_ax))
    RH_Matrix = np.zeros(shape=(s_ax, s_ax))
    if (order == 6) and (grade == 1):
        for x in range(s_ax):
            for y in range(s_ax):
                if (x-y) == 0:
                    LH_Matrix[x,y] = 1.
                elif (x-y) == 1:
                    LH_Matrix[x,y] = 1./3.
                    LH_Matrix[y,x] = 1./3.
                    RH_Matrix[x,y] = -14./18.
                    RH_Matrix[y,x] =  14./18.
                elif (x-y) == 2:
                    RH_Matrix[x,y] = -1./36.
                    RH_Matrix[y,x] = 1./36.
        LH_Matrix[0,1] = 5.
        LH_Matrix[1,0] = 1./8.
        LH_Matrix[1,2] = 3./4.
        LH_Matrix[-1,:] = LH_Matrix[0,:][::-1]
        LH_Matrix[-2,:] = LH_Matrix[1,:][::-1]

        RH_Matrix[0,0:6] = [-197./60., -5./12., 5., -5./3., 5./12., -1./20.]
        RH_Matrix[1,0:5] = [-43./96., -5./6., 9./8., 1./6., -1./96.]
        RH_Matrix[-1,:] = -RH_Matrix[0,:][::-1]
        RH_Matrix[-2,:] = -RH_Matrix[1,:][::-1]
        
        RH_Matrix *= (1./dx[axis])
        
    elif grade == 2:
        if order == 6:
            for x in range(s_ax):
                for y in range(s_ax):
                    if (x-y) == 0:
                        LH_Matrix[x,y] = 1.
                        RH_Matrix[x,y] = -2*120./97. #P6
                    elif (x-y) in [1,-1]:
                        LH_Matrix[x,y] = 12./97. #P6
                        RH_Matrix[x,y] = 120./97. #P6
                    elif (x-y) in [2,-2]:
                        LH_Matrix[x,y] = -1./194. #P6
                                #P6
            LH_Matrix[0,1:3] = [11./2., -131./4.]
            RH_Matrix[0, 0:6] = [177./16., -507./8., 783./8., -201./4., 81./16., -3./8.]

        LH_Matrix[-1,:] = LH_Matrix[0,:][::-1]
        RH_Matrix[-1,:] = RH_Matrix[0,:][::-1]

        RH_Matrix *= (1/dx[axis]**2)


    #implementing boundaries
    inv_LH_Matrix = np.linalg.inv(LH_Matrix)
    Full_Matrix = np.matmul(inv_LH_Matrix, RH_Matrix)
    derivative_matrix = Full_Matrix[1:-1,1:-1]
    dirichlet_vector = [Full_Matrix[1:-1,0], Full_Matrix[1:-1,-1]]

    return derivative_matrix, dirichlet_vector


def Create_FFTMatrix(size, dx, order = 6):
    LH_Matrix = np.zeros(shape=(size[1], size[1]), dtype=np.float64)
    RH_Matrix = np.zeros(shape=(size[1], size[1]), dtype=np.float64)
    s_ax = size[1]
    for x in range(s_ax):
        for y in range(s_ax):
            if (x-y) == 0:
                LH_Matrix[x,y] = 1.
                RH_Matrix[x,y] = -2*120./97. #P6
            elif (x-y) in [1,-1]:
                LH_Matrix[x,y] = 12./97. #P6
                RH_Matrix[x,y] = 120./97. #P6
            elif (x-y) in [2,-2]:
                LH_Matrix[x,y] = -1./194. #P6
                    #P6
    LH_Matrix[0,1:3] = [11./2., -131./4.]
    RH_Matrix[0, 0:6] = [177./16., -507./8., 783./8., -201./4., 81./16., -3./8.]

    LH_Matrix[-1,:] = LH_Matrix[0,:][::-1]
    RH_Matrix[-1,:] = RH_Matrix[0,:][::-1]

    RH_Matrix *= (1/(dx[1])**2)

    Full_Tensor = np.zeros(shape=(size[1]-2, size[1]-2, size[0]), dtype=np.float64)
    if (size[0]%2==0):
        s02 = size[0]/2
        for k in range(size[0]):
            if (k != s02):
                LH_Matrix_k = np.linalg.inv(-((2*np.pi*((k-s02)/(dx[0]*size[0])))**2)*LH_Matrix[1:-1,1:-1] + RH_Matrix[1:-1,1:-1])
                Full_Tensor[:,:,k] = np.matmul(LH_Matrix_k,LH_Matrix[1:-1,1:-1])
            else:
                Full_Tensor[:,:,k] = np.eye(size[1]-2, size[1]-2) # constant term
    else:
        print("SIZEX has to be even.")
    return Full_Tensor
