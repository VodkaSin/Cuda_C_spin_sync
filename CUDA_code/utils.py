import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gen_same_pop(k, max_det):
    #########################################
    # Return array of N_spin in k classes (uniformly distributed) and detuning distribution (k array)
    # The detuning distribution follows a standard normal distribution with classes spread across 99% (3*sigma) of [0, max_det]
    # E.g. detuning = gen_same_pop(100, 5, 50000)
    ########################################
    # pop = np.asarray([int(N_spin/k) for i in range(k)])
    det = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,k)])
    det = det[::-1]
    return det


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
    
def plot_heat(x, y, z, z_min, z_max, title, xlabel, ylabel):
    # x,y: x and y axis variables
    # z: dependent variable, dimension dim(x)*dim(y)
    # z_min, z_max: range of z to plot
    # x and ylabels supports latex
    y_rot = y[::-1]
    c = plt.pcolormesh(x, y_rot, z, cmap = 'CMRmap', vmin = z_min, vmax = z_max)
    plt.colorbar(c)
    plt.title(f'{title}',fontsize=12)
    plt.ylabel(f'{ylabel}',fontsize=12)
    plt.xlabel(f'{xlabel}',fontsize=12)


def read_results(handle):
    # Input: string (suffix of files)
    # Output: np arrays of t_store 
    result_sz_filename = f"Result_Sz_{handle}.dat"
    result_coherences_filename = f"Result_coherences_real_{handle}.dat"
    result_photon_filename = f"Result_photon_{handle}.dat"
    result_time_filename = f"Result_time_{handle}.dat"

    result_sz = np.loadtxt(result_sz_filename, dtype=np.longdouble)
    result_coherences = np.loadtxt(result_coherences_filename, dtype=np.longdouble)
    result_photon = np.loadtxt(result_photon_filename, dtype=np.longdouble)
    result_time = np.loadtxt(result_time_filename, dtype=np.longdouble)
    return result_time, result_sz, result_coherences, result_photon
