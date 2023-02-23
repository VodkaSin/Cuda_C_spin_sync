#if !defined(FUNCTIONS_H)
#define FUNCTIONS_H

//all functions
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
// random number generation
#include <curand.h>
#include <curand_kernel.h>

using namespace std;



					
// update the observables of photons 
__global__ void	calculate_photons(double tc,int num_ens,double *para_a_dev,double *para_c_dev,\
				double2 *ap_a_dev,double2 *a_dev,double2 *a_a_dev,\
				double2 *a_sp_dev,double2 *sm_dev,double2 *a_sm_dev,\
				double2 *d_ap_a_dev,double2 *d_a_dev,double2 *d_a_a_dev);

__global__ void	update_photons(double t_step, double2 *ap_a_dev,double2 *a_dev,double2 *a_a_dev,\
				double2 *d_ap_a_dev,double2 *d_a_dev,double2 *d_a_a_dev);

// update the observables of atoms 
__global__ void	calculate_atoms(double tc,int num_ens,double *para_a_dev,double *para_c_dev,\
				double2 *sz_dev,double2 *sm_dev,double2 *a_sz_dev,double2 *a_sm_dev,double2 *a_sp_dev,\
				double2 *sm_sp_dev,double2 *sm_sm_dev,double2 *sm_sz_dev, double2 *a_dev, double2 *ap_a_dev,double2 *a_a_dev,\
				double2 *d_sz_dev,double2 *d_sm_dev,double2 *d_a_sz_dev,double2 *d_a_sm_dev,double2 *d_a_sp_dev);
							
							
__global__ void	update_atoms(int num_ens,double t_step,double *para_a_dev,double2 *sz_dev,double2 *sm_dev,double2 *a_sz_dev,double2 *a_sm_dev,double2 *a_sp_dev,\
				double2 *d_sz_dev,double2 *d_sm_dev,double2 *d_a_sz_dev,double2 *d_a_sm_dev,double2 *d_a_sp_dev);

							
__global__ void	calculate_correlations(int num_ens,double t_step,double *para_a_dev,double *para_c_dev,\
					double2 *sm_sp_dev,double2 *sm_sz_dev,double2 *sm_sm_dev,double2 *sz_sz_dev,\
					double2 *a_dev,double2 *a_sm_dev,double2 *a_sp_dev,double2 *a_sz_dev,double2 *sm_dev,double2 *sz_dev,\
					double2 *d_sm_sp_dev,double2 *d_sm_sz_dev,double2 *d_sm_sm_dev,double2 *d_sz_sz_dev);
							
							
__global__ void	update_correlations(int num_ens,double t_step,double2 *sm_sp_dev,double2 *sm_sz_dev,double2 *sm_sm_dev,double2 *sz_sz_dev,\
					double2 *d_sm_sp_dev,double2 *d_sm_sz_dev,double2 *d_sm_sm_dev,double2 *d_sz_sz_dev);

#endif
