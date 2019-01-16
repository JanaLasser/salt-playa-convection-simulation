import numpy as np
from numpy import cos, sin, pi
import scipy.ndimage.filters
from scipy.special import erf

# Load initial concentration matrix from numpy-file
def Load_InitialC(path, size, frequency):
	C0 = np.load(path)
	C = np.zeros(size, dtype=np.float32) # create matrix with size
	size0 = np.shape(C0) # grid size in file
	# stretch matrix C0 from size0 to size with interpolating between the gridpoints
	for y in range(size[1]):
		yt = y/(size[1]-1)*(size0[1]-1)
		y0 = int(yt) 
		yt0 = yt - y0
		if y == 0 or y == (size[1] - 1): y1 = y0
		else: y1 = y0+ 1
		for x in range(size[0]):
			xt = ((x/size[0]*frequency)%1.)*size0[0]
			x0 = int(xt)
			xt0 = xt - x0
			x1 = x0 +1
			if x1 > size0[0] - 1: x1 = 0
			C[x,y] = C0[x0,y0]*(1.-yt0)*(1.-xt0)
			C[x,y] += C0[x1,y0]*(1.-yt0)*(xt0)
			C[x,y] += C0[x0,y1]*(yt0)*(1.-xt0)
			C[x,y] += C0[x1,y1]*(yt0)*(xt0)
	return C


# Add white noise to concentration matrix with amplitude "factor"
# apply gaussian filter with sigma = [2,2]
# make sure the seed is known and recorded. Since the way we are
# adding noise to the system calls the random number generator
# multiple times, we add +1 to the seed for every call. This way
# we still get randomness in the system but keep the seed predictable
def AddRandomNoise(C, size, seed, factor = 0.01):
	C_temp = np.zeros(size, dtype=np.float32)
	for x in range(size[0]):
		for y in range(size[1]):
			if y > 6 and y < size[1]-6:
				np.random.seed(seed)
				C_temp[x,y] += (np.random.random()*2.-1.)*factor
				seed += 1
	C_temp = scipy.ndimage.filters.gaussian_filter(C_temp, [2,2], mode = 'wrap')
	#C_temp[:,0:2] = 0. # Boundary itself is not noisy
	#C_temp[:,-3:-1] = 0.
	print(C_temp)
	return C+C_temp

# Load steady state
def Load_SteadyStateC(size, dx, length):
	C = np.zeros(size, dtype=np.float32)
	for y in range(size[1]):
		C[:,y] += np.exp(-y*dx[1])/(1-np.exp(-length[1])) + 1/(1-np.exp(+length[1]))
	return C

# Load initial distribution with a linear salinity decay from 1 at the surface to 0 at the bottom
def Load_LinearDecayC(size, dx, length):
    C = np.zeros(size, dtype=np.float32)
    H = length[1]
    dx = dx[1]
    num_gridpoints = size[1]

    decay = np.linspace(0,H,size[1])
    decay = 1. - decay / (H / 2.)
    decay = np.where(decay < 0, 0, decay)

    for y in range(num_gridpoints):
        C[:,y] += decay
        
    C = np.rot90(C)
    return C

def Load_ErrorFunction(size, dx, length):
    C = np.zeros(size, dtype=np.float32)
    H = length[1]
    dx = dx[1]
    num_gridpoints = size[1]

    decay = np.linspace(0,H,size[1])
    decay = 1. - erf(decay)

    for y in range(num_gridpoints):
        C[:,y] += decay
        
    C = np.rot90(C)
    return C


# Define matrices to have sinusoidal evaporation rate at the top boundary
def SinusoidalEvaporation(size_b, length, parameters):
    waves = parameters['waves']
    amp = parameters['amplitude']
    phi = parameters['phi']
    W = length[0]
    H = length[1]
    NW = size_b[0]
    NH = size_b[1]
    
    Omega0 = np.zeros(size_b, dtype=np.float32)
    Ux0 = np.zeros(size_b, dtype=np.float32)
    Uy0 = np.zeros(size_b, dtype=np.float32)
    
    for x in range(size_b[0]):
        for y in range(1,size_b[1]+1):
            # vorticity
            Omega0[x,y-1] = 2*waves/W*(cos(y*pi/(NH+1))+1)
            Omega0[x,y-1] += cos(y*pi/(NH+1))*W/(2*waves*H**2)
            Omega0[x,y-1] *= -amp * sin(2*pi*waves*x / NW + phi)*0.5*pi
            
            # vertical velocity field
            Uy0[x,y-1] = -1 - amp/2.*cos(2*pi*waves*x/NW+phi)*(cos(y*pi/(NH+1))+1)
            
            # horizontal velocity field
            Ux0[x,y-1] = - amp*W/(4*waves*H)*sin(2*pi*waves*x/NW+phi)*sin(y*pi/(NH+1))
            
    U0 = np.array([Ux0, Uy0])
    return (U0, Omega0)

# get grid point y from height h (in natural units)
def yy(h, size, length):
	return int(h*(size[1]-1)/length[1])

# get height h in natural units from grid point y
def h(y, size, length):
	return y*length[1]/(size[1]-1)

# Create Rayleigh-Matrix to get height dependent Rayleigh number
def Rayleigh(y, size, length, parameters):
	mu = length[1]-2.0
	sigma = 0.2
	return parameters['Ra'] +  parameters['Ra2']*0.5*(1+erf((h(y, size, length)-mu)/sigma))
def Rayleigh_Matrix(size, length, parameters):
	Ra = np.zeros(size, dtype=np.float32)
	for y in range(size[1]):
		Ra[:,y] = Rayleigh(y, size, length, parameters)
	return Ra