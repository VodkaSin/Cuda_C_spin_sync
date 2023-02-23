//all functions
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
// random number generation
#include <curand.h>
#include <curand_kernel.h>
// parameters and functions
#include "functions.cuh"

using namespace std;



// update the observables of photons 
__global__ void	calculate_photons(double tc,int num_ens,double *para_a_dev,double *para_c_dev,\
				double2 *ap_a_dev,double2 *a_dev,double2 *a_a_dev,\
				double2 *a_sp_dev,double2 *sm_dev,double2 *a_sm_dev,\
				double2 *d_ap_a_dev,double2 *d_a_dev,double2 *d_a_a_dev){
// laser pulse 
// if current time tc exceeds pulse duration then set F to 0
	if (tc > para_c_dev[5]){ 
		para_c_dev[4] = 0.;
	}

// photon number ada
// coupling 
	(*d_ap_a_dev).x = - (para_c_dev[1] + para_c_dev[2])*(*ap_a_dev).x;
	(*d_ap_a_dev).y = 0.;
	for (int i =0; i < num_ens; i++){
		(*d_ap_a_dev).x += -2.*para_a_dev[i+5*num_ens]*para_a_dev[i]*a_sp_dev[i].y;
	}
// driving 
	(*d_ap_a_dev).x += -2.*sqrtf(para_c_dev[1])*para_c_dev[4]*(cos(para_c_dev[3]*tc)*(*a_dev).y + sin(para_c_dev[3]*tc)*(*a_dev).x);


	
//  photon amplitude a
	double2 c_omega_c;
	c_omega_c.x = para_c_dev[0];
	c_omega_c.y = -0.5*(para_c_dev[1] + para_c_dev[2]);

	(*d_a_dev).x = c_omega_c.y*(*a_dev).x+c_omega_c.x*(*a_dev).y;
	(*d_a_dev).y = -(c_omega_c.x*(*a_dev).x -c_omega_c.y*(*a_dev).y); 
	
	for (int i = 0; i < num_ens; i++){
		(*d_a_dev).x += para_a_dev[i+5*num_ens]*para_a_dev[i]*sm_dev[i].y;
		(*d_a_dev).y += - para_a_dev[i+5*num_ens]*para_a_dev[i]*sm_dev[i].x;
	}	
// driving 	
	(*d_a_dev).x += sqrtf(para_c_dev[1])*para_c_dev[4]*sin(-para_c_dev[3]*tc); 
	(*d_a_dev).y += - sqrtf(para_c_dev[1])*para_c_dev[4]*cos(-para_c_dev[3]*tc);



//
// photon-photon correlation a2
	(*d_a_a_dev).x = 2.*(c_omega_c.x*(*a_a_dev).y + c_omega_c.y*(*a_a_dev).x); 
	(*d_a_a_dev).y = -2.*(c_omega_c.x*(*a_a_dev).x - c_omega_c.y*(*a_a_dev).y);
	
	for (int i = 0; i< num_ens; i++) {
		(*d_a_a_dev).x += 2.*para_a_dev[i+5*num_ens]*para_a_dev[i]*a_sm_dev[i].y;
		(*d_a_a_dev).y += - 2.*para_a_dev[i+5*num_ens]*para_a_dev[i]*a_sm_dev[i].x;
	}
// driving 	
	(*d_a_a_dev).x += 2.*sqrtf(para_c_dev[1])*para_c_dev[4]*(cos(-para_c_dev[3]*tc)*(*a_dev).y + sin(-para_c_dev[3]*tc)*(*a_dev).x); 
	(*d_a_a_dev).y += -2.*sqrtf(para_c_dev[1])*para_c_dev[4]*(cos(-para_c_dev[3]*tc)*(*a_dev).x - sin(-para_c_dev[3]*tc)*(*a_dev).y);
										  
}


__global__ void	update_photons(double t_step, double2 *ap_a_dev,double2 *a_dev,double2 *a_a_dev,\
								double2 *d_ap_a_dev,double2 *d_a_dev,double2 *d_a_a_dev){
	(*ap_a_dev).x += t_step*(*d_ap_a_dev).x;
	
	(*a_dev).x += t_step*(*d_a_dev).x;
	(*a_dev).y += t_step*(*d_a_dev).y;
	
	(*a_a_dev).x += t_step*(*d_a_a_dev).x;
	(*a_a_dev).y += t_step*(*d_a_a_dev).y;
}


// update the observables of atoms 
__global__ void	calculate_atoms(double tc,int num_ens,double *para_a_dev,double *para_c_dev,\
				double2 *sz_dev,double2 *sm_dev,double2 *a_sz_dev,double2 *a_sm_dev,double2 *a_sp_dev,\
				double2 *sm_sp_dev,double2 *sm_sm_dev,double2 *sm_sz_dev, double2 *a_dev, double2 *ap_a_dev, double2 *a_a_dev,\
				double2 *d_sz_dev,double2 *d_sm_dev,double2 *d_a_sz_dev,double2 *d_a_sm_dev,double2 *d_a_sp_dev){
// calculate the index of matrix
int i = threadIdx.x + blockIdx.x*blockDim.x;
if (i < num_ens){
// atomic population inversion 
		d_sz_dev[i].x = 4.*para_a_dev[i+5*num_ens]*a_sp_dev[i].y\
				- para_a_dev[i+2*num_ens]*(1.+sz_dev[i].x) + para_a_dev[i+3*num_ens]*(1.-sz_dev[i].x);
		d_sz_dev[i].y = 0.;
	
// atomic polarization
		double2 c_omega_a;
		c_omega_a.x = para_a_dev[i+num_ens];
		c_omega_a.y = -(0.5*(para_a_dev[i+2*num_ens]+para_a_dev[i+3*num_ens]) + para_a_dev[i+4*num_ens]);
		
		d_sm_dev[i].x  = c_omega_a.x*sm_dev[i].y + c_omega_a.y*sm_dev[i].x - para_a_dev[i+5*num_ens]*a_sz_dev[i].y;
		d_sm_dev[i].y  = -(c_omega_a.x*sm_dev[i].x - c_omega_a.y*sm_dev[i].y) + para_a_dev[i+5*num_ens]*a_sz_dev[i].x;
		
// atom-photon correlation
// a_sp 
		double2 c_omega_c;
		c_omega_c.x = 0.;
		c_omega_c.y = -0.5*(para_c_dev[1] + para_c_dev[2]);

		double a_ap_sz = 2.*(a_sz_dev[i].x*(*a_dev).x + a_sz_dev[i].y*(*a_dev).y) \
				+ (1. + (*ap_a_dev).x)*sz_dev[i].x - 2.*((*a_dev).x*(*a_dev).x + (*a_dev).y*(*a_dev).y)*sz_dev[i].x;
						
		d_a_sp_dev[i].x = -((c_omega_a.x-c_omega_c.x)*a_sp_dev[i].y+(-c_omega_a.y-c_omega_c.y)*a_sp_dev[i].x) \
					+ para_a_dev[i+5*num_ens]*(para_a_dev[i]-1.)*sm_sp_dev[i+i*num_ens].y;
						
		d_a_sp_dev[i].y = (c_omega_a.x-c_omega_c.x)*a_sp_dev[i].x - (-c_omega_a.y-c_omega_c.y)*a_sp_dev[i].y\
					- 0.5*para_a_dev[i+5*num_ens]*(1.-sz_dev[i].x) \
					- para_a_dev[i+5*num_ens]*(para_a_dev[i]-1.)*sm_sp_dev[i+i*num_ens].x \
					- para_a_dev[i+5*num_ens]*a_ap_sz;
			
		for (int j = 0; j < num_ens; j ++){
			if (j != i){
				d_a_sp_dev[i].x += para_a_dev[j+5*num_ens]*para_a_dev[j]*sm_sp_dev[j+i*num_ens].y;
				d_a_sp_dev[i].y += - para_a_dev[j+5*num_ens]*para_a_dev[j]*sm_sp_dev[j+i*num_ens].x;
			}
		}

// driving 	
		d_a_sp_dev[i].x += sqrtf(para_c_dev[1])*para_c_dev[4]*(-cos(-para_c_dev[3]*tc)*sm_dev[i].y + sin(-para_c_dev[3]*tc)*sm_dev[i].x);
		d_a_sp_dev[i].y += - sqrtf(para_c_dev[1])*para_c_dev[4]*(cos(-para_c_dev[3]*tc)*sm_dev[i].x + sin(-para_c_dev[3]*tc)*sm_dev[i].y);

					
// a_sm
		double2 a_a_sz;
		a_a_sz.x = 2.*((*a_dev).x*a_sz_dev[i].x - (*a_dev).y*a_sz_dev[i].y) \
				+ (*a_a_dev).x*sz_dev[i].x - 2.*((*a_dev).x*(*a_dev).x -(*a_dev).y*(*a_dev).y)*sz_dev[i].x;
		a_a_sz.y = 2.*((*a_dev).y*a_sz_dev[i].x + (*a_dev).x*a_sz_dev[i].y) \
				+ (*a_a_dev).y*sz_dev[i].x - 2.*(2.*(*a_dev).x*(*a_dev).y)*sz_dev[i].x;

		d_a_sm_dev[i].x = (c_omega_c.x + c_omega_a.x)*a_sm_dev[i].y +  (c_omega_c.y + c_omega_a.y)*a_sm_dev[i].x\
					+ para_a_dev[i+5*num_ens]*((para_a_dev[i]-1.)*sm_sm_dev[i+i*num_ens].y - a_a_sz.y);
		
		d_a_sm_dev[i].y = -((c_omega_c.x + c_omega_a.x)*a_sm_dev[i].x- (c_omega_c.y + c_omega_a.y)*a_sm_dev[i].y) \
					- para_a_dev[i+5*num_ens]*((para_a_dev[i]-1.)*sm_sm_dev[i+i*num_ens].x - a_a_sz.x); 		
		
		for (int j = 0; j < num_ens; j ++){
			if (j != i){
				d_a_sm_dev[i].x += para_a_dev[j+5*num_ens]*para_a_dev[j]*sm_sm_dev[j+i*num_ens].y;
				d_a_sm_dev[i].y += -para_a_dev[j+5*num_ens]*para_a_dev[j]*sm_sm_dev[j+i*num_ens].x;
			}
		}

// driving 
		d_a_sm_dev[i].x += sqrtf(para_c_dev[1])*para_c_dev[4]*(sin(-para_c_dev[3]*tc)*sm_dev[i].x + cos(-para_c_dev[3]*tc)*sm_dev[i].y);
		d_a_sm_dev[i].y += - sqrtf(para_c_dev[1])*para_c_dev[4]*(cos(-para_c_dev[3]*tc)*sm_dev[i].x - sin(-para_c_dev[3]*tc)*sm_dev[i].y);


// a_sz
		double2 a_ap_sm,a_a_sp;

		a_a_sp.x = (*a_a_dev).x*sm_dev[i].x + (*a_a_dev).y*sm_dev[i].y + 2.*((*a_dev).x*a_sp_dev[i].x- (*a_dev).y*a_sp_dev[i].y) \
			-2.*(((*a_dev).x*(*a_dev).x - (*a_dev).y*(*a_dev).y)*sm_dev[i].x  + 2.*(*a_dev).x*(*a_dev).y*sm_dev[i].y);
		a_a_sp.y = -(*a_a_dev).x*sm_dev[i].y + (*a_a_dev).y*sm_dev[i].x + 2.*((*a_dev).x*a_sp_dev[i].y + (*a_dev).y*a_sp_dev[i].x)\
			-2.*(-((*a_dev).x*(*a_dev).x - (*a_dev).y*(*a_dev).y)*sm_dev[i].y + 2.*(*a_dev).x*(*a_dev).y*sm_dev[i].x);

		a_ap_sm.x = a_sm_dev[i].x*(*a_dev).x + a_sm_dev[i].y*(*a_dev).y + (1. + (*ap_a_dev).x)*sm_dev[i].x \
			+ (*a_dev).x*a_sp_dev[i].x + (*a_dev).y*a_sp_dev[i].y - 2.*((*a_dev).x*(*a_dev).x + (*a_dev).y*(*a_dev).y)*sm_dev[i].x;
		a_ap_sm.y = a_sm_dev[i].y*(*a_dev).x - a_sm_dev[i].x*(*a_dev).y + (1. + (*ap_a_dev).x)*sm_dev[i].y \
			+ (*a_dev).y*a_sp_dev[i].x - (*a_dev).x*a_sp_dev[i].y - 2.*((*a_dev).x*(*a_dev).x + (*a_dev).y*(*a_dev).y)*sm_dev[i].y;
	
		d_a_sz_dev[i].x = c_omega_c.x*a_sz_dev[i].y + c_omega_c.y*a_sz_dev[i].x \
				- para_a_dev[i+2*num_ens]*((*a_dev).x +a_sz_dev[i].x) + para_a_dev[i+3*num_ens]*((*a_dev).x-a_sz_dev[i].x)\
				+ para_a_dev[i+5*num_ens]*(sm_dev[i].y + (para_a_dev[i]-1.)*sm_sz_dev[i+i*num_ens].y) \
				+ 2.*para_a_dev[i+5*num_ens]*(a_a_sp.y - a_ap_sm.y);
				
		d_a_sz_dev[i].y = -(c_omega_c.x*a_sz_dev[i].x - c_omega_c.y*a_sz_dev[i].y) \
				- para_a_dev[i+2*num_ens]*((*a_dev).y + a_sz_dev[i].y) + para_a_dev[i+3*num_ens]*((*a_dev).y - a_sz_dev[i].y)\
				- para_a_dev[i+5*num_ens]*(sm_dev[i].x + (para_a_dev[i]-1.)*sm_sz_dev[i+i*num_ens].x) \
				- 2.*para_a_dev[i+5*num_ens]*(a_a_sp.x - a_ap_sm.x);
				

		for (int j = 0; j < num_ens; j ++){
			if (j != i){
				d_a_sz_dev[i].x += para_a_dev[j+5*num_ens]*para_a_dev[j]*sm_sz_dev[j+i*num_ens].y;
				d_a_sz_dev[i].y += -para_a_dev[j+5*num_ens]*para_a_dev[j]*sm_sz_dev[j+i*num_ens].x;
			}
		}
// driving 		
		d_a_sz_dev[i].x += sqrtf(para_c_dev[1])*para_c_dev[4]*sin(-para_c_dev[3]*tc)*sz_dev[i].x;
		d_a_sz_dev[i].y += -sqrtf(para_c_dev[1])*para_c_dev[4]*cos(-para_c_dev[3]*tc)*sz_dev[i].x;

}							
}

__global__ void	update_atoms(int num_ens,double t_step,double *para_a_dev,double2 *sz_dev,double2 *sm_dev,double2 *a_sz_dev,double2 *a_sm_dev,double2 *a_sp_dev,\
				double2 *d_sz_dev,double2 *d_sm_dev,double2 *d_a_sz_dev,double2 *d_a_sm_dev,double2 *d_a_sp_dev){
// calculate the index of matrix
int i = threadIdx.x + blockIdx.x*blockDim.x;
if (i < num_ens){
// atom loss
	para_a_dev[i] += -t_step*para_a_dev[i + 6*num_ens]*para_a_dev[i];
	
	sz_dev[i].x += t_step*d_sz_dev[i].x;
	
	sm_dev[i].x += t_step*d_sm_dev[i].x;
	sm_dev[i].y += t_step*d_sm_dev[i].y;
	
	a_sz_dev[i].x += t_step*d_a_sz_dev[i].x;
	a_sz_dev[i].y += t_step*d_a_sz_dev[i].y;
	
	a_sm_dev[i].x += t_step*d_a_sm_dev[i].x;
	a_sm_dev[i].y += t_step*d_a_sm_dev[i].y;
	
	a_sp_dev[i].x += t_step*d_a_sp_dev[i].x;
	a_sp_dev[i].y += t_step*d_a_sp_dev[i].y;
}
}



__global__ void	calculate_correlations(int num_ens,double t_step,double *para_a_dev,double *para_c_dev,\
					double2 *sm_sp_dev,double2 *sm_sz_dev,double2 *sm_sm_dev,double2 *sz_sz_dev,\
					double2 *a_dev,double2 *a_sm_dev,double2 *a_sp_dev,double2 *a_sz_dev,double2 *sm_dev,double2 *sz_dev,\
					double2 *d_sm_sp_dev,double2 *d_sm_sz_dev,double2 *d_sm_sm_dev,double2 *d_sz_sz_dev){
int i = threadIdx.x;
int j = blockIdx.x;
if (i < num_ens){
	if (j < num_ens) {
//
// atom-atom correlation
		double2 c_omega_a_i,c_omega_a_j;
		c_omega_a_i.x = para_a_dev[i+num_ens];
		c_omega_a_i.y = -(0.5*(para_a_dev[i+2*num_ens]+para_a_dev[i+3*num_ens]) + para_a_dev[i+4*num_ens]);
		c_omega_a_j.x = para_a_dev[j+num_ens];
		c_omega_a_j.y = -(0.5*(para_a_dev[j+2*num_ens]+para_a_dev[j+3*num_ens]) + para_a_dev[j+4*num_ens]);

// sm_sp
		double2 a_sz_sp,ap_sm_sz;
		a_sz_sp.x = (*a_dev).x*sm_sz_dev[i + j*num_ens].x + (*a_dev).y*sm_sz_dev[i + j*num_ens].y + a_sz_dev[j].x*sm_dev[i].x + a_sz_dev[j].y*sm_dev[i].y \
			+ sz_dev[j].x*a_sp_dev[i].x - 2.*sz_dev[j].x*((*a_dev).x*sm_dev[i].x + (*a_dev).y*sm_dev[i].y);
		a_sz_sp.y = (*a_dev).y*sm_sz_dev[i + j*num_ens].x - (*a_dev).x*sm_sz_dev[i + j*num_ens].y + a_sz_dev[j].y*sm_dev[i].x - a_sz_dev[j].x*sm_dev[i].y \
			+ sz_dev[j].x*a_sp_dev[i].y - 2.*sz_dev[j].x*((*a_dev).y*sm_dev[i].x - (*a_dev).x*sm_dev[i].y);
		
		ap_sm_sz.x = (*a_dev).x*sm_sz_dev[j + i*num_ens].x + (*a_dev).y*sm_sz_dev[j + i*num_ens].y + sm_dev[j].x*a_sz_dev[i].x + sm_dev[j].y*a_sz_dev[i].y\
			+ a_sp_dev[j].x*sz_dev[i].x -2.*((*a_dev).x*sm_dev[j].x + (*a_dev).y*sm_dev[j].y)*sz_dev[i].x;
		ap_sm_sz.y = -(*a_dev).y*sm_sz_dev[j + i*num_ens].x + (*a_dev).x*sm_sz_dev[j + i*num_ens].y + sm_dev[j].y*a_sz_dev[i].x - sm_dev[j].x*a_sz_dev[i].y\
			- a_sp_dev[j].y*sz_dev[i].x -2.*(-(*a_dev).y*sm_dev[j].x + (*a_dev).x*sm_dev[j].y)*sz_dev[i].x;

		d_sm_sp_dev[j + i*num_ens].x = ((c_omega_a_j.x - c_omega_a_i.x)*sm_sp_dev[j + i*num_ens].y + (c_omega_a_j.y + c_omega_a_i.y)*sm_sp_dev[j + i*num_ens].x) \
						- para_a_dev[j+5*num_ens]*a_sz_sp.y + para_a_dev[i+5*num_ens]*ap_sm_sz.y;
		d_sm_sp_dev[j + i*num_ens].y = - ((c_omega_a_j.x - c_omega_a_i.x)*sm_sp_dev[j + i*num_ens].x - (c_omega_a_j.y + c_omega_a_i.y)*sm_sp_dev[j + i*num_ens].y) \
						+ para_a_dev[j+5*num_ens]*a_sz_sp.x - para_a_dev[i+5*num_ens]*ap_sm_sz.x;
																
// sm_sz
		double2 a_sz_sz,a_sm_sp,ap_sm_sm;
		a_sz_sz.x = a_sz_dev[j].x*sz_dev[i].x + (*a_dev).x*sz_sz_dev[j + i*num_ens].x +sz_dev[j].x*a_sz_dev[i].x -2.*(*a_dev).x*sz_dev[j].x*sz_dev[i].x;
		a_sz_sz.y = a_sz_dev[j].y*sz_dev[i].x + (*a_dev).y*sz_sz_dev[j + i*num_ens].x +sz_dev[j].x*a_sz_dev[i].y -2.*(*a_dev).y*sz_dev[j].x*sz_dev[i].x;
		
		a_sm_sp.x = (*a_dev).x*sm_sp_dev[j + i*num_ens].x - (*a_dev).y*sm_sp_dev[j + i*num_ens].y + sm_dev[j].x*a_sp_dev[i].x - sm_dev[j].y*a_sp_dev[i].y\
			+ a_sm_dev[j].x*sm_dev[i].x + a_sm_dev[j].y*sm_dev[i].y\
			-2.*((*a_dev).x*(sm_dev[j].x*sm_dev[i].x + sm_dev[j].y*sm_dev[i].y) - (*a_dev).y*(sm_dev[j].y*sm_dev[i].x - sm_dev[j].x*sm_dev[i].y));
		a_sm_sp.y = (*a_dev).y*sm_sp_dev[j + i*num_ens].x + (*a_dev).x*sm_sp_dev[j + i*num_ens].y + sm_dev[j].y*a_sp_dev[i].x + sm_dev[j].x*a_sp_dev[i].y \
			+ a_sm_dev[j].y*sm_dev[i].x - a_sm_dev[j].x*sm_dev[i].y  \
			-2.*((*a_dev).y*(sm_dev[j].x*sm_dev[i].x + sm_dev[j].y*sm_dev[i].y) + (*a_dev).x*(sm_dev[j].y*sm_dev[i].x - sm_dev[j].x*sm_dev[i].y));
		
		ap_sm_sm.x = (*a_dev).x*sm_sm_dev[j + i*num_ens].x + (*a_dev).y*sm_sm_dev[j + i*num_ens].y + sm_dev[j].x*a_sp_dev[i].x  + sm_dev[j].y*a_sp_dev[i].y \
			+ a_sp_dev[j].x*sm_dev[i].x + a_sp_dev[j].y*sm_dev[i].y \
			-2.*((*a_dev).x*(sm_dev[j].x*sm_dev[i].x - sm_dev[j].y*sm_dev[i].y) + (*a_dev).y*(sm_dev[j].x*sm_dev[i].y + sm_dev[j].y*sm_dev[i].x));
		ap_sm_sm.y = -(*a_dev).y*sm_sm_dev[j + i*num_ens].x + (*a_dev).x*sm_sm_dev[j + i*num_ens].y + sm_dev[j].y*a_sp_dev[i].x  - sm_dev[j].x*a_sp_dev[i].y \
			- a_sp_dev[j].y*sm_dev[i].x + a_sp_dev[j].x*sm_dev[i].y \
			-2.*(-(*a_dev).y*(sm_dev[j].x*sm_dev[i].x - sm_dev[j].y*sm_dev[i].y) + (*a_dev).x*(sm_dev[j].x*sm_dev[i].y + sm_dev[j].y*sm_dev[i].x));

		d_sm_sz_dev[j + i*num_ens].x = c_omega_a_j.x*sm_sz_dev[j + i*num_ens].y +  c_omega_a_j.y*sm_sz_dev[j + i*num_ens].x\
				- para_a_dev[j+5*num_ens]*a_sz_sz.y + 2.*para_a_dev[i+5*num_ens]*(a_sm_sp.y - ap_sm_sm.y)\
				- para_a_dev[i+2*num_ens]*(sm_dev[j].x + sm_sz_dev[j + i*num_ens].x) + para_a_dev[i+3*num_ens]*(sm_dev[j].x - sm_sz_dev[j + i*num_ens].x);													
		d_sm_sz_dev[j + i*num_ens].y = -(c_omega_a_j.x*sm_sz_dev[j + i*num_ens].x - c_omega_a_j.y*sm_sz_dev[j + i*num_ens].y)\
				+ para_a_dev[j+5*num_ens]*a_sz_sz.x - 2.*para_a_dev[i+5*num_ens]*(a_sm_sp.x - ap_sm_sm.x)\
				- para_a_dev[i+2*num_ens]*(sm_dev[j].y+sm_sz_dev[j + i*num_ens].y) + para_a_dev[i+3*num_ens]*(sm_dev[j].y - sm_sz_dev[j + i*num_ens].y);
// sm_sm
		double2 a_sm_sz,a_sz_sm;
		a_sm_sz.x = (*a_dev).x*sm_sz_dev[j + i*num_ens].x - (*a_dev).y*sm_sz_dev[j + i*num_ens].y + sm_dev[j].x*a_sz_dev[i].x - sm_dev[j].y*a_sz_dev[i].y\
				  + a_sm_dev[j].x*sz_dev[i].x -2.*((*a_dev).x*sm_dev[j].x - (*a_dev).y*sm_dev[j].y)*sz_dev[i].x;
		a_sm_sz.y = (*a_dev).y*sm_sz_dev[j + i*num_ens].x + (*a_dev).x*sm_sz_dev[j + i*num_ens].y + sm_dev[j].y*a_sz_dev[i].x + sm_dev[j].x*a_sz_dev[i].y\
				  + a_sm_dev[j].y*sz_dev[i].x -2.*((*a_dev).y*sm_dev[j].x + (*a_dev).x*sm_dev[j].y)*sz_dev[i].x;
				  
		a_sz_sm.x = (*a_dev).x*sm_sz_dev[i + j*num_ens].x - (*a_dev).y*sm_sz_dev[i + j*num_ens].y + a_sz_dev[j].x*sm_dev[i].x - a_sz_dev[j].y*sm_dev[i].y\
				  + sz_dev[j].x*a_sm_dev[i].x -2.*sz_dev[j].x*((*a_dev).x*sm_dev[i].x - (*a_dev).y*sm_dev[i].y);
		a_sz_sm.y = (*a_dev).y*sm_sz_dev[i + j*num_ens].x + (*a_dev).x*sm_sz_dev[i + j*num_ens].y + a_sz_dev[j].y*sm_dev[i].x + a_sz_dev[j].x*sm_dev[i].y\
				  + sz_dev[j].x*a_sm_dev[i].y -2.*sz_dev[j].x*((*a_dev).y*sm_dev[i].x + (*a_dev).x*sm_dev[i].y);
		
		d_sm_sm_dev[j + i*num_ens].x = ((c_omega_a_j.y + c_omega_a_i.y)*sm_sm_dev[j + i*num_ens].x + (c_omega_a_j.x + c_omega_a_i.x)*sm_sm_dev[j + i*num_ens].y) \
					- para_a_dev[i+5*num_ens]*a_sm_sz.y - para_a_dev[j+5*num_ens]*a_sz_sm.y;
		d_sm_sm_dev[j + i*num_ens].y = - ((c_omega_a_j.x + c_omega_a_i.x)*sm_sm_dev[j + i*num_ens].x - (c_omega_a_j.y + c_omega_a_i.y)*sm_sm_dev[j + i*num_ens].y) \
					+ para_a_dev[i+5*num_ens]*a_sm_sz.x + para_a_dev[j+5*num_ens]*a_sz_sm.x;

// sz_sz							
		double2 a_sp_sz;
		
		a_sp_sz.x = (*a_dev).x*sm_sz_dev[j + i*num_ens].x + (*a_dev).y*sm_sz_dev[j + i*num_ens].y + sm_dev[j].x*a_sz_dev[i].x + sm_dev[j].y*a_sz_dev[i].y \
			+ a_sp_dev[j].x*sz_dev[i].x - 2.*sz_dev[i].x*((*a_dev).x*sm_dev[j].x + (*a_dev).y*sm_dev[j].y);
		a_sp_sz.y = (*a_dev).y*sm_sz_dev[j + i*num_ens].x - (*a_dev).x*sm_sz_dev[j + i*num_ens].y - sm_dev[j].y*a_sz_dev[i].x + sm_dev[j].x*a_sz_dev[i].y \
			+ a_sp_dev[j].y*sz_dev[i].x - 2.*sz_dev[i].x*((*a_dev).y*sm_dev[j].x - (*a_dev).x*sm_dev[j].y);

		d_sz_sz_dev[j + i*num_ens].x = 4.*para_a_dev[i+5*num_ens]*a_sz_sp.y + 4.*para_a_dev[j+5*num_ens]*a_sp_sz.y \
					- para_a_dev[i+2*num_ens]*(sz_dev[j].x + sz_sz_dev[j + i*num_ens].x) + para_a_dev[i+3*num_ens]*(sz_dev[j].x-sz_sz_dev[j + i*num_ens].x) \
					- para_a_dev[j+2*num_ens]*(sz_dev[i].x + sz_sz_dev[j + i*num_ens].x) + para_a_dev[j+3*num_ens]*(sz_dev[i].x-sz_sz_dev[j + i*num_ens].x);
		d_sz_sz_dev[j + i*num_ens].y = 0.;		
	}
}															
}

__global__ void	update_correlations(int num_ens,double t_step,double2 *sm_sp_dev,double2 *sm_sz_dev,double2 *sm_sm_dev,double2 *sz_sz_dev,\
				double2 *d_sm_sp_dev,double2 *d_sm_sz_dev,double2 *d_sm_sm_dev,double2 *d_sz_sz_dev){
int i = threadIdx.x;
int j = blockIdx.x;
if (i < num_ens){
	if (j < num_ens){
		sm_sp_dev[j + i*num_ens].x += t_step*d_sm_sp_dev[j + i*num_ens].x; 
		sm_sp_dev[j + i*num_ens].y += t_step*d_sm_sp_dev[j + i*num_ens].y; 
		
		sm_sz_dev[j + i*num_ens].x += t_step*d_sm_sz_dev[j + i*num_ens].x;
		sm_sz_dev[j + i*num_ens].y += t_step*d_sm_sz_dev[j + i*num_ens].y; 
		
		sm_sm_dev[j + i*num_ens].x += t_step*d_sm_sm_dev[j + i*num_ens].x;
		sm_sm_dev[j + i*num_ens].y += t_step*d_sm_sm_dev[j + i*num_ens].y; 
		
		sz_sz_dev[j + i*num_ens].x += t_step*d_sz_sz_dev[j + i*num_ens].x;
		sz_sz_dev[j + i*num_ens].y += t_step*d_sz_sz_dev[j + i*num_ens].y; 
	}
}
}
