import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import re
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm


def gen_same_pop(k, max_det, pos=None):
    #########################################
    # Return the detuning array from small to large
    # Arguments: 
    #           k: number of classes intended
    #           max_det: 3 sigma for generating the distribution
    #           pos: Bool, True for only generating positive detunings, False splits half for negative
    # E.g. detuning = gen_same_pop(100, 5, 50000)
    ########################################
    # pop = np.asarray([int(N_spin/k) for i in range(k)])
    if pos == None:
        det = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,k)])
        return det[::-1]
    elif k%2 == 0:
        # Even number of classes:
        if pos == True:
            det_pos = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,int(k/2))])
            det_neg = -det_pos
            return np.concatenate((det_neg, det_pos[::-1]), axis=None)
        if pos == False:
            det_pos = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,int(k/2))])
            det_neg = det_pos
        return np.concatenate((det_neg, det_pos[::-1]), axis=None)
    else: 
        # Odd number of classes, set the middle one to be 0
        if pos == True:
            det_pos = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,int((k-1)/2))])
            det_neg = -det_pos
            return np.concatenate((det_neg, np.zeros(1), det_pos[::-1]), axis=None)
        if pos == False:
            det_pos = max_det * np.asarray([2*(1-norm.cdf(i)) for i in np.linspace(0,3,int((k-1)/2))])
            det_neg = det_pos
            return np.concatenate((det_neg, np.zeros(1), det_pos[::-1]), axis=None)

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
    if np.min(sz)>0.0:
        return -1
    while i<len(t_list):
        if sz[i]>0.0:
            i+=1
        else:
            return t_list[i]

def findT0(sz, t_list):
    #########################################
    # Returns the first time index that sz goes below -0.95
    # Returns -1 if sz does not go below -0.95
    # Loop through sz to find the first value <= -0.95
    #########################################
    i=0
    if np.min(sz)>-0.95:
        return -1
    while i<len(t_list):
        if sz[i]>-0.95:
            i+=1
        else:
            return i    
    

def crit_1(handle, tol):
    #########################################
    # Input file handle, reads files
    # Compare |Td(1)-Td(0)|> = < 0.05 Td(0)
    # > (not fullfilled) returns -1
    # < (fullfilled) returns 0
    # = (exactly fullfilled) returns 1
    #########################################
    results = read_results(handle)
    tlist = results[0]
    sz0 = results[1][:,0]
    Td_0 = findTd(sz0, tlist)
    sz1 = results[1][:,1]
    Td_1 = findTd(sz1, tlist)
    diff = np.abs(Td_1-Td_0) - 0.05*Td_0
    if diff > 0:
        return -1, diff
    elif np.abs(diff) < tol:
        return 0, diff
    else:
        return 1, diff
    
def crit_2(handle, tol):
    #########################################
    # Input file handle, reads files
    # Return True if when sz0 first crosses -0.95, 
    # sz1 is less than -0.95 + tol. False otherwise
    #########################################
    results = read_results(handle)
    tlist = results[0]
    sz0 = results[1][:,0]
    T0_0 = findT0(sz0, tlist)
    if T0_0 == -1:
        print("Warning: sz0 does not go below -0.95")
    sz1 = results[1][:,1][T0_0]
    if sz1 > tol-0.95:
        return False
    else:
        return True

def Wt(sz, w0=None, detuning=None):
    #########################################
    # Returns the total energy given sz and detuninsg
    # sz: (e.g.)
    # array([[ 1.       ,  1.       ],
    #        [ 1.       ,  1.       ],
    #        [ 1.       ,  1.       ],
    #        ...,
    #        [-0.9986385,  0.9982722],
    #        [-0.9986399,  0.9982717],
    #        [-0.9986413,  0.9982712]], dtype=float64)
    # detuning: array([  0, 140])
    # Formula:
    #   W(t)=sum w*(sz+1/2) (hbar=1)
    #########################################
    if detuning != None and w0!= None:
        w = w0 + detuning
    else:
        w = 1
    sz_shift = sz + 0.5
    wsz_shift = w * sz_shift
    wsz_sum = np.sum(wsz_shift,axis=1)
    return wsz_sum
    
def diffentiate(dt, func):
    #########################################
    # Returns the differential of func, size -1
    #########################################
    return np.diff(func)/dt
    
    
def cut_time(t_list, endtime):
    #########################################
    # Returns the index of t_list when endtime is reached
    #########################################
    if endtime<0:
        return -1
    return int(np.size(t_list)*endtime/t_list[-1])
    
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


def runcmd(command):
    #########################################
    # Runs the given command and its options as a subprocess.
    # It redirects and intercepts all stdout and stderr 
    # so that it can process and decides what to print depending on the line.
    # For e.g if the stdout line is the progress bar from the command,
    # it will use `tqdm` library to show it a pretty progress bar in our jupyter cell output
    #
    # This came from a need to show the progress bar for a simulation run 
    # but the jupyter cell output don't like it very much. 
    ########################################
    t = None        # `t` will be the progress bar object
    prev_pv = -1    # `prev_pv` will contain previous iteration's progress bar value (pv)
    
    # Run the command as a subprocess
    process = Popen(command, stdout=PIPE, shell=False, stderr=STDOUT, bufsize=1, close_fds=True, universal_newlines=True)
    
    # Iterate through each line and decide what to do with it
    for line in iter(process.stdout.readline, b''):
        
        # If line is empty, exit loop (we assume that's when command has completed)
        if len(line) == 0:
            break
        
        # Check if we got the 'progress' line (e.g " 13% [|||||||        ]")
        progress = re.findall(r"^\s*(\d+)%\s*\[.*\].*", line)
        if (len(progress) > 0):
            # Get the current progress value (pv)
            pv = int(progress[0])
            
            # We only update the progress bar if its value is different from the prev iterations
            if pv > prev_pv and pv < 100:
                # Initialize progress bar
                if t is None:
                    bar_format = '{percentage:3.0f}% [{bar} ]'
                    t = tqdm(total=100, bar_format=bar_format, colour='green')
                
                # Manually update the progress bar by passing in the increment
                t.update(pv-prev_pv)
                
                # Force refresh the bar
                t.refresh()
                
                # Store the current line pv
                prev_pv = pv
        else:
            # Print line
            print(line, end="") 
    
    # Clean up
    if t is not None:
        t.close()
    process.stdout.close()
    process.wait()