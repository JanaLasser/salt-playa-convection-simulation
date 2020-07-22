# salt-playa-convection-simulation
Two-dimensional simulation code for buoyancy driven convection below an evaporating salt lake.  

**author**: Jana Lasser  
**copyright** Copyright 2020, Geophysical pattern formation in salt playa  
**credits** Jana Lasser, Marcel Ernst  
**version** 1.0.0  

The simulation code was used to investigate the behaviour of convecting salty water in the sand below and evaporating salt lake that is suspected to cause the formation of polygonal salt crust patterns on the surface. The connection to salt crust papers is described in [_Salt polygons are caused by convection_, J. Lasser et al. 2020](https://arxiv.org/pdf/1902.03600). The properties of the convective dynamics is expored in more detail in [_Stability of convection in dry salt lakes_, M. Ernst, J. Lasser, L. Goehring 2020](https://arxiv.org/abs/2004.10578).


A detailed description of the theory behind the simulation code and the implementation is given in [Marcel Ernst's master's thesis](http://hdl.handle.net/21.11116/0000-0002-16A7-9). A description of the connection of buoyancy driven porous media convection to pattern formation in salt deserts is diven in [Jana Lasser's PhD thesis](http://hdl.handle.net/11858/00-1735-0000-002E-E5DB-2).

## Usage
The script ```main.py``` allows for input of different simulation parameters specifying the simulation domain and boundary conditions over the command line (see also main.py -h for all available options).

### Parameters
* ```-dest [PATH]``` Absolute or relative path to the folder where the simulation output will be saved to (default: current script location)
* ```-Ra [FLOAT]``` Rayleigh number of the system (default: 100)
* ```-H [INT]``` Domain height in natural units (default: 10)
* ```-W [INT]``` Domain width in natural units (default: 10)
* ```-res [INT]``` Spatial resolution of the uniform simulation point grid in grid points per natural unit length (default: 6)
* ```-T [FLOAT]``` Maximum simulation time in natural units (default: 25)
* ```-savetime [FLOAT]``` Interval at which snapshots of the simulation will be saved to disk in the form of binary files containing numpy arrays (default: 0.1)
* ```-clf [FLOAT]``` CLF-constant for the adaptive time-stepping routine (default 0.05)
* ```-waves [INT]``` Number of minima in the sinusioidal modulation of the evaporation rate at the top boundary (default: 1). Has no effect if ```amplitude=0```
* ```-amplitude [FLOAT]``` Amplitude of the sinusoidal modulation of the evaporation rate at the top boundary (default: 0)
* ```-plt (flag)``` Specifies whether or not plots of the simulation (default: False)
* ```-saveall (flag)``` Whether or not all fields (salinity, velocity in z-direction, velocity in x-direction) should be saved (default: only salinity field is saved)
* ```-init [std, dyn]``` Type of salinity distribution to be used as initial condition. ```std``` uses the steady state solution of the governing equations, ```dyn``` uses an exponential decay function with a length scale dependent on the Rayleigh number of the system (default: ```std```)
* ```-S [INT]``` Seed for the random number generator that is used to generate random noise on the initial salinity distribution (default: seed will be generated based on system time)

### Examples
#### Simulation with specified domain size and Rayleigh number
To start a simulation at $Ra=500$ with a simulation domain size of $40\,$L$\times 100\,$L and a resolution of 8 gridpoints per unit length:  

```python main.py -Ra 500 -W 40 -H 100 -res 8 -plt -saveall```  

This will create a new folder in your current directory with the simulation parameters encoded in the folder name. The folder contains three subfolders for each of the three simulated fields (salinity, velocity in x-direction, velocity in z-direction) as well as a text-file with the simulation parameters. The simulation will save snapshots of all three fields (```-saveall```) in their respective folders at time intervals of ```dt=0.1``` (default value) and also save plots of the fields alongside the matrices (```-plt```). The simulation will run for 25 natural time units (default value) use the steady state solution of the governing equations as initial condition and use a CLF constant of 0.05 (default values).

#### Simulation with top boundary modulation
To start a simulation with a pre-defined seed (for example because you want to test different parameters at the exact same initial conditions) and a top boundary condition that is modulated by 4 sinusoidal waves with an amplitude of 0.5:  

``` python main.py -waves 4 -amplitude 0.5 -S 12345 -dest testing/```  

This will start a simulation at $Ra=100$ with a domain size of $10\,$L$\times 10\,$L and resolution of 6 grid points per unit length (default values). Only the salinity field will be saved every $0.1\,$T (default value), this time to a custom directory ```testing/``` (```-dest```). The simulation will use the random seed 12345 specified by you (```-S```).
