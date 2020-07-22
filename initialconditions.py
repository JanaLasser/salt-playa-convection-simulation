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
	return C+C_temp

# Load steady state
def Load_SteadyStateC(size, dx, length):
	C = np.zeros(size, dtype=np.float32)
	for y in range(size[1]):
		C[:,y] += np.exp(-y*dx[1])/(1-np.exp(-length[1])) + 1/(1-np.exp(+length[1]))
	return C

# length scales measured from fits to the salinity decay of the convecting system
# see plot 4.5 (b) of thesis http://hdl.handle.net/11858/00-1735-0000-002E-E5DB-2
# these length scales are hardcoded. 
decay_length_scales ={
20:0.74926090497, #interpolated
23:0.6758709087350034,
25:0.62694424457, #interpolated
28:0.5535542483107837,
30:0.53720645221, #interpolated
33:0.5126847580359059,
35:0.48241579293, #interpolated
38:0.43701234526894506,
40:0.4328093432823327,
60:0.38249406911536826,
80:0.2914565557322542,
100:0.24326839206963757,
120:0.22345835730519928,
140:0.20496248898707617,
160:0.17531070436450336,
180:0.15560449777130897,
200:0.14343490232045025,
220:0.12565691532110695,
230:0.12474404119173808,
240:0.12101668656721586,
250:0.11347251335736594,
260:0.11448933091076345,
270:0.1068533983537615,
280:0.10042433560869075,
290:0.09852570635307657,
300:0.09971496345394001,
400:0.07705545682258624,
500:0.06231201291500437,
600:0.04878597931141859,
700:0.04172401582818638,
800:0.03773896603660117,
900:0.03331292105753958,
1000:0.029152661038141778,
1100:0.02658871416754643,
1200:0.025390683253352146,
1300:0.023629313499373198,
1400:0.0213701942804863,
1500:0.019274195912062243,
1600:0.018795444312516927,
1700:0.016958691209834127,
1800:0.015973254171916303,
1900:0.01498781714, #interpolated
2000:0.01400238011, #interpolated
3000:0.01400238011, #interpolated
4000:0.01400238011} #interpolated

def Load_DynamicDecayC(size, dx, length, Ra):
	C = np.zeros(size, dtype=np.float32)
	H = length[1]
	L = decay_length_scales[Ra]
	num_gridpoints = size[1]

	decay = np.linspace(0,H,size[1])
	decay = np.exp(-decay/L)

	for y in range(num_gridpoints):
	    C[:,y] += decay
	    
	C = np.rot90(C)
	return C

def Load_SpecificDecayC(size, dx, length, decay_length_scale):
	C = np.zeros(size, dtype=np.float32)
	H = length[1]
	L = decay_length_scale
	num_gridpoints = size[1]

	decay = np.linspace(0,H,size[1])
	decay = np.exp(-decay/L)

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
            Uy0[x,y-1] = -1 + amp/2.*cos(2*pi*waves*x/NW+phi)*(cos(y*pi/(NH+1))+1)
            
            # horizontal velocity field
            Ux0[x,y-1] = + amp*W/(4*waves*H)*sin(2*pi*waves*x/NW+phi)*sin(y*pi/(NH+1))
            
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