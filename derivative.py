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

# obsolete
'''
def CreateFFTMatrix(input, size, dx, axis = 0):
	left_coeff, right_coeff = input[:]
	size_axis = size[axis]
	left_matrix = np.zeros(shape=(size_axis,size_axis))
	right_matrix = np.zeros(shape=(size_axis,size_axis))
	for i0 in range(size_axis):
			#if i0 >= size_axis/2:
			#	i = i0 - size_axis
			#else: 
			#	i = i0
		i = i0 - size_axis/2
		i = i*2.*np.pi/(dx[axis]*size_axis)
		for j0 in range(size_axis):
			if (i0-j0) == 0:
				left_matrix[i0,j0] = -(i**2) *left_coeff[0]- 2./(dx[axis]**2)*(right_coeff[2]/9. + right_coeff[1]/4. + right_coeff[0]/1.)
				right_matrix[i0,j0] = left_coeff[0]
			elif (i0-j0)%size_axis in [1, size_axis - 1]:
				left_matrix[i0,j0] = right_coeff[0]/(dx[axis]**2) - (i**2) * left_coeff[1]
				right_matrix[i0,j0] = left_coeff[1]
			elif (i0-j0)%size_axis in [2, size_axis - 2]:
				left_matrix[i0,j0] = right_coeff[1]/(4.*dx[axis]**2) - (i**2)*left_coeff[2]
				right_matrix[i0,j0] = left_coeff[2]
			elif (i0-j0)%size_axis in [3, size_axis - 3]:
				left_matrix[i0,j0] = right_coeff[2]/(9.*dx[axis]**2)

	inv_left_matrix = np.linalg.inv(left_matrix)
	derivative_matrix = np.matmul(inv_left_matrix,right_matrix)
	return derivative_matrix
	'''
	
