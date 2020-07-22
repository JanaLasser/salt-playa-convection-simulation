### periodic derivative matrices ###

import numpy as np

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
