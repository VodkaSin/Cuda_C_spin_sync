import numpy as np
import matplotlib.pyplot as plt

def unit_time(gk, N_spin, kappa):
    #########################################
    # Calculates the unit time
    # Notion refers to Haroche Superradiance
    #########################################
    Omega_0 = 2*gk
    lim = Omega_0*np.sqrt(N_spin)
    if kappa < 1.5*lim:
        print("Warning: not in the overdamped regime")
    Gc = Omega_0**2/kappa
    Tr = 1/Gc/N_spin
    return Tr


def delay_time(gk, N_spin, kappa):
    #########################################
    # Calculates the delay time
    # Approximates the time <sz> reaches 0
    # Estimator of the width of superradiance pulse
    # When N_spin > 10000, use log to approximate
    # Error <= 7%
    #########################################
    Tr = unit_time(gk, N_spin, kappa)
    if N_spin > 10000:
        Td =  Tr * np.log(N_spin)
    else:
        Td = Tr * np.sum([1/(1+i) for i in range(N_spin)])
    return Td


def findTd(sz,t_list):
    #########################################
    # Returns the delay time estimated from the sz simulation results
    # Returns -1 if sz does not go below 0.0
    # Loop through sz to find the first value <= 0.0
    #########################################
  i=0
  if sz[-1]>0.0:
    return -1
  while i<len(t_list):
    if sz[i]>0.0:
        i+=1
    else:
        return t_list[i]
    
def plot_heat(x, y, z, z_min, z_max):
    # x,y: x and y axis variables
    # z: dependent variable, dimension dim(x)*dim(y)
    # z_min, z_max: range of z to plot
    c = plt.pcolormesh(x, y, z, cmap = 'CMRmap', vmin = z_min, vmax = z_max)
    plt.colorbar(c)
    plt.title('$<\sigma_z>$',fontsize=18)
    plt.ylabel('Detuning',fontsize=18)
    plt.xlabel('Time$(\mu s)$',fontsize=18)