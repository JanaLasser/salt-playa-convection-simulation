# non-periodic

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

	elif (order == 6) and (grade == 2):
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
		LH_Matrix[-1,:] = LH_Matrix[0,:][::-1]
		RH_Matrix[0, 0:6] = [177./16., -507./8., 783./8., -201./4., 81./16., -3./8.]
		RH_Matrix[-1,:] = RH_Matrix[0,:][::-1]
		RH_Matrix *= (1/dx[axis]**2)

	#implementing boundaries
	inv_LH_Matrix = np.linalg.inv(LH_Matrix)
	Full_Matrix = np.matmul(inv_LH_Matrix, RH_Matrix)
	derivative_matrix = Full_Matrix[1:-1,1:-1]
	dirichlet_vector = [Full_Matrix[1:-1,0], Full_Matrix[1:-1,-1]]
	return derivative_matrix, dirichlet_vector


def Create_FFTMatrix(size, dx):
	LH_Matrix = np.zeros(shape=(size[1], size[1]))
	RH_Matrix = np.zeros(shape=(size[1], size[1]))
	for x in range(size[1]):
		for y in range(size[1]):
			if (x-y) == 0:
				LH_Matrix[x,y] = 1.
				RH_Matrix[x,y] = -2*120./97. #P6
			elif (x-y) in [1,-1]:
				LH_Matrix[x,y] = 12/97. #P6
				RH_Matrix[x,y] = 120./97. #P6
			elif (x-y) in [2,-2]:
				LH_Matrix[x,y] = -1./194. #P6
	LH_Matrix[0,1:3] = [11./2., -131./4.]
	LH_Matrix[-1,:] = LH_Matrix[0,:][::-1]
	RH_Matrix[0, 0:6] = [177./16., -507./8., 783./8., -201./4., 81./16., -3./8.]
	RH_Matrix[-1,:] = RH_Matrix[0,:][::-1]
	RH_Matrix *= (1/(dx[1])**2)

	Full_Tensor = np.zeros(shape=(size[1], size[1], size[0]))
	if (size[0]%2==0):
		s02 = size[0]/2
		for k in range(size[0]):
			if (k != s02):
				LH_Matrix_k = np.linalg.inv(-((2*np.pi*((k-s02)/(dx[0]*(size[0]) )))**2)*LH_Matrix+RH_Matrix)
				Full_Tensor[:,:,k] = np.matmul(LH_Matrix_k, LH_Matrix)
			else:
				Full_Tensor[:,:,k] = np.eye(size[1], size[1])
	else:
		print("SIZEX has to be even.")
	return Full_Tensor[1:-1,1:-1,:]