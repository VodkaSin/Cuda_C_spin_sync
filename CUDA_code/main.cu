#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <ctime>
#include <iostream>
#include <curand_kernel.h>
#include <curand.h>

#ifdef __unix__
#include <unistd.h>
#elif defined(_WIN32)|| defined(WIN32) 
#include <stdint.h>
#endif

#include "functions.cuh"
using namespace std;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define PI (4.0 * atan(1.0));
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Wrapper for if anything goes wrong with GPU
// e.g. gpuAssert(cudaMalloc((void**)&para_a_dev,6*num_ens*sizeof(double)));
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
int loadDoubleData(char* filename, double* out, int out_size);
void printProgress(int curr, int total);

/**
Arguments
POSITION	VARNAME		TYPE	NOTES
---------------------------------------------
argv[1]		num_ens: 	int		Number of classes
argv[2]		N_total: 	int		Total number of spins
argv[3]		theta_0: 	float	Coefficient of PI
argv[4]		phi_0: 		float	Coefficient of PI
argv[5]		coup_a_0:	float	Cavity-atom coupling strength
argv[6]		gamma_a_0:	float	Atom decay rate
argv[7]		chi_a_0:	float	Atom dephase rate
argv[8]		kappa_c_0:	float	Cavity decay rate
argv[9]		t_max:		float	Simulation end time
argv[10]	t_num:		int		Number of steps
argv[11]	handle: 	string	File handle to save

Example run:
file.exe 2 100000 1.0 0.0 1.6 0.0 0.0 160.0 2.0 60000 60000

To compile:
nvcc -w functions.cu main.cu -o file

To compile and run in one line
cls && nvcc -w functions.cu main.cu -o file && file.exe 2 100000 1.0 0.0 1.6 0.0 0.0 160.0 2.0 60000 60000

*/
int main(int argc, char** argv) {

	// Print input values
	printf("num_ens:\t\t\t%s\n", argv[1]);
	printf("N_total:\t\t\t%s\n", argv[2]);
	printf("theta_0:\t\t\t%s\n", argv[3]);
	printf("phi_0:\t\t\t\t%s\n", argv[4]);
	printf("coup_a_0:\t\t\t%s\n", argv[5]);
	printf("gamma_a_0 (atom decay):\t\t%s\n", argv[6]);
	printf("chi_a_0 (atom dephase):\t\t%s\n", argv[7]);
	printf("kappa_c_0 (cavity decay):\t%s\n", argv[8]);
	printf("t_max:\t\t\t\t%s\n", argv[9]);
	printf("t_num:\t\t\t\t%s\n", argv[10]);
	printf("\n");

	//************************************************************************** INITIAL PARAM *********************************
	// Ensemble settings
	int num_ens = atoi(argv[1]); 	// Number of classes
	int N_total = atoi(argv[2]); 	// Number of spins
	int ens_size = N_total/num_ens; // Number of spins in each class (uniform distribution)

	// Initial state
		// CSS = cos(theta/2)|g> + sin(theta/2)e^i*phi |e>
		// theta_0 = PI fully excited, theta_0 = 0 fully grounded
	double theta_0 = atof(argv[3])*PI;
	double phi_0 = atof(argv[4])*PI; 

	// System settings
		// Unit in kHz * 2pi
	double coup_a_0 =  atof(argv[5]); 	// Atom-cavity coupling
	double gamma_a_0 = atof(argv[6]); 	// Atom decay rate: [lower_a]
	double chi_a_0 =   atof(argv[7]); 	// Atom dephase rate: [sz]
	double kappa_c_0 = atof(argv[8]); 	// Cavity decay rate: [a]
	double loss_0 =    0.0;				// Atom loss (population decreases rate)
	double omega_c =   0.0; 			// Cavity detuning
	double kappa_1_c = kappa_c_0/2.0;	// Left mirror decay
	double kappa_2_c = kappa_c_0/2.0;	// Right mirror decay
	double eta_a_0 =   0.0;				// Atom pumping: [sp]

	//************************************************************************************** TIME CONSTANTS ********************************
	double t_max = atof(argv[9]);					// End of simulation
	int t_num = atoi(argv[10]);						// Total number of steps
	double t_step = t_max/t_num;					// dt size of each step
	int t_store_num = 20000;						// Total number of stored (CPU) time steps
	int t_store =  t_num/t_store_num;				// Number of steps on GPU before storing on CPU
	
	// Total number of steps > stored steps or exit
	if (t_num < t_store_num) {
		printf("[invalid param] Specify a 't_num' larger than or equal to %i", t_store_num);
		exit(EXIT_FAILURE);
		return;
	}

	// File handle
	char* handle = argv[11];

	//*************************************************************************************** PARAMETERS FOR SQUARE PULSE ******************

	double omega_d = 0.2;						// FREQUENCY OF SQUARE PULSE FOR INITIALIZATION
	//double coup_d =  0.0;					        // AMPLITUDE OF THE PULSE
	double coup_d = 0.0*3;					// AMPLITUDE OF THE PULSE
	double t_stop = 0.0*15; 					// LENGTH OF SQUARE PULSE in us
	//1.943*1.0E-7			


	//*************************************************************************************** INITIALIZE ENSEMBLES (CLASSES) ***************

	// double N_a[num_ens],omega_a[num_ens],gamma_a[num_ens],\
			eta_a[num_ens],chi_a[num_ens],coup_a[num_ens],loss_a[num_ens];
	double* N_a = new double[num_ens];
	double* omega_a = new double[num_ens];
	double* gamma_a = new double[num_ens];
	double* eta_a = new double[num_ens];
	double* chi_a = new double[num_ens];
	double* coup_a = new double[num_ens];
	double* loss_a = new double[num_ens];

	for (int i =0; i < num_ens; i++){
		N_a[i] = ens_size;
		// omega_a[i] = 10; // omega_a is loaded from Detuning.dat instead; see below.
		gamma_a[i] = gamma_a_0;
		eta_a[i] = eta_a_0;
		chi_a[i] = chi_a_0;
		coup_a[i] = coup_a_0;
		loss_a[i] = loss_0;
	}

	// Load detuning data from "Detuning.dat" into `omega_a` array.
	// Please ensure that Detuning.dat file has the same number of expected values (values_count=num_ens)
	int res = loadDoubleData("Detuning.dat", omega_a, num_ens);
	if (res != 0) {
		printf("Error: failed to load detuning data\n");
		exit(EXIT_FAILURE);
	}

	// Print out loaded data
	// printf("Loaded detuning data into omega_a:\n");
	// for (int i=0; i<num_ens; i++) {
	// 	printf("%f\n", omega_a[i]);
	// }

	// Parameters related to the atoms concatenated
	// E.g. num_ens = 3, para_a = [det1, det2, det3, dec1, dec2, dec3, ...]
	double* para_a = new double[7*num_ens];

	for  (int i = 0; i < num_ens; i++){
		para_a[i] = N_a[i];
		para_a[i+num_ens] = omega_a[i];		// Detuning
		para_a[i+2*num_ens] = gamma_a[i];	// Decay
		para_a[i+3*num_ens] = eta_a[i];		// Pump
		para_a[i+4*num_ens] = chi_a[i];		// Dephase
		para_a[i+5*num_ens] = coup_a[i];	// Coupling
		para_a[i+6*num_ens] = loss_a[i];
	}

	// Copy the parameters into the memory in GPU
	// Changed allocation size from 6 to 7

	double *para_a_dev;
	cudaMalloc((void**)&para_a_dev,7*num_ens*sizeof(double)); 
	cudaMemcpy(para_a_dev,para_a,7*num_ens*sizeof(double),cudaMemcpyHostToDevice);


	//************************************************************************************** INITIALIZE COHERENT SPIN STATE ***************

	double* theta = new double[num_ens];
	double* phi = new double[num_ens];

	for (int i=0; i < num_ens; i++){
		theta[i] = theta_0;
		phi[i] = phi_0;
	}

	// CSS = cos(theta/2)|g> + sin(theta/2)e^i*phi |e>
	// cu coefficient for the excited state
	// cl coefficient for the ground state

	double2* cu = new double2[num_ens];
	double2* cl = new double2[num_ens];

	for (int i=0; i< num_ens; i++){
		cu[i].x = sin(0.5*theta[i])*cos(phi[i]);
		cu[i].y = sin(0.5*theta[i])*sin(phi[i]);
		
		cl[i].x = cos(0.5*theta[i]); 
		cl[i].y = 0.; 
	}

	double para_c[9];
	para_c[0] = omega_c;
	para_c[1] = kappa_1_c;
	para_c[2] = kappa_2_c;

	para_c[3] = omega_d;
	para_c[4] = coup_d;
	para_c[5] = t_stop;



	double *para_c_dev;
	cudaMalloc((void**)&para_c_dev,9*sizeof(double));
	cudaMemcpy(para_c_dev,para_c,9*sizeof(double),cudaMemcpyHostToDevice);


	double *t_step_dev;
	cudaMalloc((void**)&t_step_dev,sizeof(double));
	cudaMemcpy(t_step_dev,&t_step,sizeof(double),cudaMemcpyHostToDevice);



	// on CPU side 
	double2 ap_a,a,a_a;
	// double2 sz[num_ens],sm[num_ens],a_sz[num_ens],a_sm[num_ens],a_sp[num_ens];
	// double2 sm_sp[num_ens*num_ens],sm_sz[num_ens*num_ens],\
		sm_sm[num_ens*num_ens],sz_sz[num_ens*num_ens];

	double2* sz = new double2[num_ens];
	double2* sm = new double2[num_ens];
	double2* a_sz = new double2[num_ens];
	double2* a_sm = new double2[num_ens];
	double2* a_sp = new double2[num_ens];
	double2* sm_sp = new double2[num_ens*num_ens];
	double2* sm_sz = new double2[num_ens*num_ens];
	double2* sm_sm = new double2[num_ens*num_ens];
	double2* sz_sz = new double2[num_ens*num_ens];

	// for initial values 
	double2 sm_1,sp_1,sz_1,sm_2,sz_2; 

	//************************************************************************************** INITIALIZE OBSERVABLES ***************
	ap_a.x = 0.; ap_a.y = 0.; a.x = 0.; a.y = 0.; a_a.x =0.; a_a.y = 0.; 

	for (int i= 0; i < num_ens; i++){
		sz_1.x = (cu[i].x*cu[i].x + cu[i].y*cu[i].y) - (cl[i].x*cl[i].x + cl[i].y*cl[i].y); 
		sz_1.y = 0.; 
		sm_1.x = cu[i].x*cl[i].x + cu[i].y*cl[i].y;
		sm_1.y = -cu[i].x*cl[i].y + cu[i].y*cl[i].x; 
		sp_1.x = cu[i].x*cl[i].x + cu[i].y*cl[i].y;
		sp_1.y = cu[i].x*cl[i].y - cu[i].y*cl[i].x;
		
		sz[i].x = sz_1.x; sz[i].y = sz_1.y;
		sm[i].x = sm_1.x; sm[i].y = sm_1.y; 
		
		a_sp[i].x = 0.; a_sp[i].y = 0.;
		a_sz[i].x = 0.; a_sz[i].y = 0.;
		a_sm[i].x = 0.; a_sm[i].y = 0.; 
		
		for (int j = 0; j < num_ens; j++){
			sz_2.x = (cu[j].x*cu[j].x + cu[j].y*cu[j].y) - (cl[j].x*cl[j].x + cl[j].y*cl[j].y); 
			sz_2.y = 0.; 
			sm_2.x = cu[j].x*cl[j].x + cu[j].y*cl[j].y;
			sm_2.y = -cu[j].x*cl[j].y + cu[j].y*cl[j].x; 
			
			sm_sp[j + i*num_ens].x = sm_2.x*sp_1.x - sm_2.y*sp_1.y; 
			sm_sp[j + i*num_ens].y = sm_2.x*sp_1.y + sm_2.y*sp_1.x; 
			
			sm_sz[j + i*num_ens].x = sm_2.x*sz_1.x - sm_2.y*sz_1.y;
			sm_sz[j + i*num_ens].y = sm_2.x*sz_1.y + sm_2.y*sz_1.x;
			
			sm_sm[j + i*num_ens].x = sm_2.x*sm_1.x - sm_2.y*sm_1.y;
			sm_sm[j + i*num_ens].y = sm_2.x*sm_1.y + sm_2.y*sm_1.x;
					
			sz_sz[j + i*num_ens].x = sz_2.x*sz_1.x - sz_2.y*sz_1.y;
			sz_sz[j + i*num_ens].y = sz_2.x*sz_1.y + sz_2.y*sz_1.x;	
		}
	}

	// on GUP side 
	double2 *ap_a_dev,*a_dev,*a_a_dev;
	double2 *sz_dev,*sm_dev,*a_sz_dev,*a_sm_dev,*a_sp_dev;
	double2 *sm_sp_dev,*sm_sz_dev,*sm_sm_dev,*sz_sz_dev;

	// create observables on GPU side 
	cudaMalloc((void**)&ap_a_dev,sizeof(double2));
	cudaMalloc((void**)&a_dev,sizeof(double2));
	cudaMalloc((void**)&a_a_dev,sizeof(double2));

	cudaMalloc((void**)&sz_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&sm_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&a_sz_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&a_sm_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&a_sp_dev,num_ens*sizeof(double2));

	cudaMalloc((void**)&sm_sp_dev,num_ens*num_ens*sizeof(double2));
	cudaMalloc((void**)&sm_sz_dev,num_ens*num_ens*sizeof(double2));
	cudaMalloc((void**)&sm_sm_dev,num_ens*num_ens*sizeof(double2));
	cudaMalloc((void**)&sz_sz_dev,num_ens*num_ens*sizeof(double2));



	// copy observables on GPU side 
	cudaMemcpy(ap_a_dev,&ap_a,sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(a_dev,&a,sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(a_a_dev,&a_a,sizeof(double2),cudaMemcpyHostToDevice);

	cudaMemcpy(sz_dev,sz,num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(sm_dev,sm,num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(a_sz_dev,a_sz,num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(a_sm_dev,a_sm,num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(a_sp_dev,a_sp,num_ens*sizeof(double2),cudaMemcpyHostToDevice);

	cudaMemcpy(sm_sp_dev,sm_sp,num_ens*num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(sm_sz_dev,sm_sz,num_ens*num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(sm_sm_dev,sm_sm,num_ens*num_ens*sizeof(double2),cudaMemcpyHostToDevice);
	cudaMemcpy(sz_sz_dev,sz_sz,num_ens*num_ens*sizeof(double2),cudaMemcpyHostToDevice);

	//***************
	// derivatives 
	double2 *d_ap_a_dev,*d_a_dev,*d_a_a_dev;
	double2 *d_sz_dev,*d_sm_dev,*d_a_sz_dev,*d_a_sm_dev,*d_a_sp_dev;
	double2 *d_sm_sp_dev,*d_sm_sz_dev,*d_sm_sm_dev,*d_sz_sz_dev;

	// create observables on GPU side 
	cudaMalloc((void**)&d_ap_a_dev,sizeof(double2));
	cudaMalloc((void**)&d_a_dev,sizeof(double2));
	cudaMalloc((void**)&d_a_a_dev,sizeof(double2));

	cudaMalloc((void**)&d_sz_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&d_sm_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&d_a_sz_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&d_a_sm_dev,num_ens*sizeof(double2));
	cudaMalloc((void**)&d_a_sp_dev,num_ens*sizeof(double2));

	cudaMalloc((void**)&d_sm_sp_dev,num_ens*num_ens*sizeof(double2));
	cudaMalloc((void**)&d_sm_sz_dev,num_ens*num_ens*sizeof(double2));
	cudaMalloc((void**)&d_sm_sm_dev,num_ens*num_ens*sizeof(double2));
	cudaMalloc((void**)&d_sz_sz_dev,num_ens*num_ens*sizeof(double2));

	FILE *Result_time, *Result_Sz, *Result_photon, *Result_coherences_real;
	
	// time of simulation
	//************************************************************************** OPEN FILE *****************************************
		// Warning: handle must not be longer than 60 characters = keep it short
		// Space is allowed, just enclose with 
	char fname1[100];
	char fname2[100];
	char fname3[100];
	char fname4[100];


	snprintf(fname1, 100, "Result_time_%s.dat", handle);
	Result_time = fopen(fname1,"w");
	
	
	snprintf(fname2, 100, "Result_Sz_%s.dat", handle);
	Result_Sz = fopen(fname2,"w");

	snprintf(fname3, 100, "Result_photon_%s.dat", handle);
	Result_photon= fopen(fname3,"w");

	snprintf(fname4, 100, "Result_coherences_real_%s.dat", handle);
	Result_coherences_real= fopen(fname4,"w");


	// ***********************************
	// simulations starts
	// ***********************************
	clock_t start_clock, end_clock;
	start_clock = clock();
	double tc;

	// `force_exit` will be used to abruptly stop the simulation under certain conditions,
	// for e.g:
	// - when the calculated `sz` is NaN.
	bool force_exit = false;

	// update the old reduced density matrix 
	for (int t = 1; t < t_num; t++){
		
		printProgress(t, t_num-1); // t_num-1 because loop would never go to t_num
		
		// printf("t %i of t_num %i, tc %1f \n", t, t_num, tc);
		tc = t*t_step;
		// printf("tc %1f \n", tc);

		//************************************
		// calculate derivatives 

		// calculate the photon observables
		// ap_a, a, a_a 
		calculate_photons<<<1,1>>>(tc,num_ens,para_a_dev,para_c_dev,\
					ap_a_dev,a_dev,a_a_dev,\
					a_sp_dev,sm_dev,a_sm_dev,\
					d_ap_a_dev,d_a_dev,d_a_a_dev);
		cudaThreadSynchronize();

		// calculate the atomic observables and atom-photon correlations
		// sz, sm, a_sz, a_sm, a_sp 
		calculate_atoms<<<1,num_ens>>>(tc,num_ens,para_a_dev,para_c_dev,\
						sz_dev,sm_dev,a_sz_dev,a_sm_dev,a_sp_dev,\
						sm_sp_dev,sm_sm_dev,sm_sz_dev,a_dev,ap_a_dev,a_a_dev,\
						d_sz_dev,d_sm_dev,d_a_sz_dev,d_a_sm_dev,d_a_sp_dev);
		cudaThreadSynchronize();

		// calculate the atom-atom correlations 
		// sm_sp, sm_sz, sm_sm, sz_sz
		calculate_correlations<<<num_ens,num_ens>>>(num_ens,t_step,para_a_dev,para_c_dev,\
							sm_sp_dev,sm_sz_dev,sm_sm_dev,sz_sz_dev,\
							a_dev,a_sm_dev,a_sp_dev,a_sz_dev,sm_dev,sz_dev,\
							d_sm_sp_dev,d_sm_sz_dev,d_sm_sm_dev,d_sz_sz_dev);
		cudaThreadSynchronize();

		//*************************************
		// update observables

		update_photons<<<1,1>>>(t_step,ap_a_dev,a_dev,a_a_dev,\
					d_ap_a_dev,d_a_dev,d_a_a_dev);
		cudaThreadSynchronize();


		update_atoms<<<1,num_ens>>>(num_ens,t_step,para_a_dev,sz_dev,sm_dev,a_sz_dev,a_sm_dev,a_sp_dev,\
					d_sz_dev,d_sm_dev,d_a_sz_dev,d_a_sm_dev,d_a_sp_dev);
		cudaThreadSynchronize();
		
		update_correlations<<<num_ens,num_ens>>>(num_ens,t_step,sm_sp_dev,sm_sz_dev,sm_sm_dev,sz_sz_dev,\
							d_sm_sp_dev,d_sm_sz_dev,d_sm_sm_dev,d_sz_sz_dev);
		cudaThreadSynchronize();

		if ( t%t_store == 0) {
		// copy the calculate observables back to CPU side 


			cudaMemcpy(sz,sz_dev,num_ens*sizeof(double2),cudaMemcpyDeviceToHost);
			cudaMemcpy(sm,sm_dev,num_ens*sizeof(double2),cudaMemcpyDeviceToHost);
			cudaMemcpy(sm_sp,sm_sp_dev,num_ens*sizeof(double2),cudaMemcpyDeviceToHost);
			cudaMemcpy(&ap_a,ap_a_dev,sizeof(double2),cudaMemcpyDeviceToHost);
			cudaMemcpy(&a,a_dev,sizeof(double2),cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();

		// store the file
			fprintf(Result_time,"%e \n",(double)t*t_step);
			fprintf(Result_photon,"%e \n",ap_a.x);
			//printf("%1f	%e	\n",tc, ap_a.x);
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			for (int i = 0; i < num_ens; i++) {
				// If the returned sz value is NaN, forcefully stop the simulation.
				if (isnan(sz[i].x)) {
					fflush(stdout);
					printf("\nError: sz[i].x is NaN, stopping simulation. (t=%i, i=%i)\n", t, i);
					force_exit = true;
					break;
				}
				
				fprintf(Result_Sz,"%e ",sz[i].x);
				fprintf(Result_coherences_real,"%e ",sm_sp[i].x); //changed from [i] to [i*num_ens] to print diagonal
			}
			fprintf(Result_Sz,"\n");
			fprintf(Result_coherences_real,"\n");
			
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
		if (force_exit) {
			printf("Abruptly stopping simulation...\n");
			break;
		}
	}

	// close the files
	fclose(Result_time);
	fclose(Result_Sz);
	fclose(Result_photon);
	fclose(Result_coherences_real);


	// close the memories 

	cudaFree(para_a_dev); cudaFree(para_c_dev); cudaFree(t_step_dev);

	cudaFree(ap_a_dev); cudaFree(a_dev); cudaFree(a_a_dev);

	cudaFree(sz_dev); cudaFree(sm_dev); cudaFree(a_sz_dev);
	cudaFree(a_sm_dev);cudaFree(a_sp_dev);

	cudaFree(sm_sp_dev); cudaFree(sm_sz_dev);
	cudaFree(sm_sm_dev); cudaFree(sz_sz_dev);

	cudaFree(d_ap_a_dev); cudaFree(d_a_dev); cudaFree(d_a_a_dev);

	cudaFree(d_sz_dev); cudaFree(d_sm_dev); cudaFree(d_a_sz_dev);
	cudaFree(d_a_sm_dev);cudaFree(d_a_sp_dev);

	cudaFree(d_sm_sp_dev); cudaFree(d_sm_sz_dev);
	cudaFree(d_sm_sm_dev); cudaFree(d_sz_sz_dev);

	delete[] N_a;
	delete[] omega_a;
	delete[] gamma_a;
	delete[] eta_a;
	delete[] chi_a;
	delete[] coup_a;
	delete[] loss_a;
	delete[] para_a;
	delete[] theta;
	delete[] phi;
	delete[] cu;
	delete[] cl;
	delete[] sz;
	delete[] sm;
	delete[] a_sz;
	delete[] a_sm;
	delete[] a_sp;
	delete[] sm_sp;
	delete[] sm_sz;
	delete[] sm_sm;
	delete[] sz_sz;

	end_clock = clock();
	// fprintf(stderr,"Program takes about %.2f s\n",(double)(ct1-ct0)/(double)CLOCKS_PER_SEC);
	printf("\n");
	printf("Program takes about %.2f s\n",(double)(end_clock - start_clock)/(double)CLOCKS_PER_SEC);
	return 0;
}

// Loads 2D matrix data from file (for e.g "Detuning.dat") 
// and assign it into given `out` 1D double array.
//
// Arguments:
// filename		char*		Name of data file. Can be relative.
// out			double* 	Pointer to output array that will contain the loaded data.
// size_out		int			Size of output array. 
//							The function will only read the same number of values as the given `size_out` number.
// 							It MUST be the same as the size of the given `out` array. (Or out-of-bounds error may occur).
//							If the size_out > number of values (rows*cols) in the file, the content of `out` array is not guaranteed.
//							If the size_out < number of values in the file, extraneous values will be truncated/will not be read.
// 
// The function expects the file to be in the following format:
// - Each row is separated by a newline '\n'.
// - Each row should not be more than 4096 characters.
// - Each column is separated by a tab '\t'. 
// - A column can have 1 or more tab-separated values.
// - Empty rows will be skipped.
//
// Returns int
// - 0 if successful
// - 1 if file not found
//
// Example of how to load detuning data into a MxN sized matrix using `loadDoubleData()`
// int M = 5; // The number of rows
// int N = 2; // The number of cols
// double* inhomo_test = new double[M*N]; // Initialize the array that we want to load the data into (use 1d array)
// loadDoubleData("Detuning_matrix_test.dat", inhomo_test, M*N); // Load detuning data
// printf("Loaded test detuning data as matrix:\n");
// for (int i=0; i<M; i++) {
// 	for (int j=0; j<N; j++) {
// 		printf("%f\t", inhomo_test[i*N+j]);
// 	}
// 	printf("\n");
// }
int loadDoubleData(char* filename, double* out, int out_size) {

	// File pointer to data file
	FILE* fp;

	// Open file 
	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("[loadDoubleData] Error: File (%s) does not exists\n", filename);
		// Return with error (1: file not found)
		return 1;
	}

    const char col_delim[] = "\t"; 	// Column delimiter
	char row[4096];					// This will store each fget attempt, might not be the entire line if the line exceeds buffer size
	char* col;						// This will store each tab-delimited value in the line
	int idx = 0;					// Keeps count of how many rows*columns we've parsed

	// printf("[loadDoubleData] Reading data file: %s\n", filename);
	while (fgets(row, sizeof(row), fp)) {
		// TODO: check if the line read is complete (ends with newline)

		// Remove trailing newline
		row[strcspn(row, "\n")] = 0;

		// Split line by delimiter (tab)
		col = strtok(row, col_delim);
		while(col != NULL) {
			if (idx >= out_size) {
				// only read up to `out_size` number of values
				break;
			}
			// Convert string to double value
			double val = strtod(col, NULL);

			// Add value to the out array
			out[idx++] = val;
			
			// Continue tokenizing the rest of the string
			col = strtok(NULL, col_delim);
		}
	}
	// Close file
	fclose(fp);
	
	// No error
	return 0;
}

// Print progress bar, given the current progress value and its total progress
// For e.g for curr=10 and total=100, the bar will show 10%
// Uses fflush(), \r, and (%.*s, %*s) format specifiers
void printProgress(int curr, int total) {
	float percentage = 100.0 * curr / total;
    int val = (int)percentage;
    int lpad = (int)(percentage/100 * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}
