#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <ctime>
#include <iostream>
using namespace std;
#include <curand_kernel.h>
#include <curand.h>
#include "functions.cuh"
#define PI (4.0 * atan(1.0));
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int main(int argc, char** argv) {
int num_ens = atoi(argv[1]); // number of classes
int p=50; // need this for symmetric freq dist // num_ens=2p+1
int N_total = atoi(argv[2]); // number of spins

clock_t ct0,ct1;
ct0 = clock();
// double axe[p+1];
double* axe = new double[p+1];  // SGK
int ens_size = N_total/num_ens;
double phi_0 = atof(argv[4])*PI;
// double inhomo[num_ens];
double* inhomo = new double[num_ens];  // SGK


double b= 0.999999;
double aa= 0.0000000001;
int n= p+1;
double c = (b - aa)/(n - 1);
for(int i = 0; i < n; ++i){
axe[i] =  aa + i*c;
//printf("%f \n",axe[i]);
}
axe[n - 1] = b;

double maxdetun = 500;
double sigma = 0.022;
double sqrthalf = 0.707;



// Normal distribution
// Writing detuning values in a file 
FILE *Detuning;
Detuning = fopen("Detuning.dat","w");

if(num_ens==1){
	double freq = 0;
inhomo[0] = freq; //always start at zero detuning
fprintf(Detuning,"%e \n",inhomo[0]);
fclose(Detuning);
}

//if(num_ens>1){
//	for (int i=1; i<p+1; i++){
//		double x= axe[i];
//		double tt1, tt2, lnx, sgn;
//		sgn = (x < 0) ? -1.0f : 1.0f;
//		x = (1 - x)*(1 + x);    
//		lnx = log(x);
//		tt1 = (2/(3.1416*0.147) + 0.5f * lnx);
//		tt2 = 1/(0.147) * lnx;
//		//inhomo[p+i] =  sgn*sqrt(-tt1 + sqrtf(tt1*tt1 - tt2))*maxdetun/3.1416;
//		//inhomo[p-i] = -sgn*sqrt(-tt1 + sqrtf(tt1*tt1 - tt2))*maxdetun/3.1416; 
//		inhomo[i-1] = ((0.5 * erfc(-axe[i] * sqrthalf/ sigma) - 0.5)*2-1)*maxdetun;
//		inhomo[2*p+1-i] = -((0.5 * erfc(-axe[i] * sqrthalf/ sigma) - 0.5)*2-1)*maxdetun;
//		//printf("%1f\n", inhomo[i+1]);
//	}
//	inhomo[p]=0;
//	for(int i=0;i<num_ens;i++){
//		fprintf(Detuning,"%e \n",inhomo[i]);
//	}
//	fclose(Detuning);
//}
if(num_ens==5){
double delta = atof(argv[3]);
double delta1 = atof(argv[4]);
int i0 = 0; int i1 = 1; int i2=2; int i3=3; int i4=4;
inhomo[i0]= -delta1;
fprintf(Detuning,"%e \n",inhomo[0]);
inhomo[i1]= -delta;
fprintf(Detuning,"%e \n",inhomo[1]);
inhomo[i2]= delta-delta;
fprintf(Detuning,"%e \n",inhomo[2]);
inhomo[i3]= delta;
fprintf(Detuning,"%e \n",inhomo[3]);
inhomo[i4]= delta1;
fprintf(Detuning,"%e \n",inhomo[4]);
fclose(Detuning);
}

if(num_ens==3){
double delta = atof(argv[3]);
int i0 = 0; int i1 = 1; int i2=2;
inhomo[i0]= -delta;
fprintf(Detuning,"%e \n",inhomo[0]);
inhomo[i1]= delta-delta;
fprintf(Detuning,"%e \n",inhomo[1]);
inhomo[i2]= delta;
fprintf(Detuning,"%e \n",inhomo[2]);
fclose(Detuning);
}

if(num_ens==2){
double delta = atof(argv[3]);
int i0 = 0; int i1 = 1;
inhomo[i0]= -delta;
fprintf(Detuning,"%e \n",inhomo[0]);
inhomo[i1]= delta-delta;
fprintf(Detuning,"%e \n",inhomo[1]);
fclose(Detuning);
}



//********************************************************************************************* PARAMETERS AND DEFINITIONS ********************************************************
// Unit in kHz * 2pi
double gamma_a_0 = 0.0*2.0;				// ATOM DECAY
double eta_a_0 =   0.0;						// ATOM PUMPING
double chi_a_0 =   0.0;						// ATOM DEPHASING
double coup_a_0 =  atof(argv[9]);					// ATOM CAVITY COUPLING
double loss_0 =    0.0;						// ATOM LOSS
double kappa_1_c = 1.0*100.0;				// LEFT MIRROR DECAY
double kappa_2_c = 1.0*100.0;				// RIGHT MIRROR DECAY

double theta_0 = atof(argv[6])*PI; 						// INTIAL STATE...... PI FOR EXCITED STATE AND (PI*0.5) FOR EQUAL SUPERPOSITION
double zero_phonon= 0.0;					// ZERO PHONON LINE
double omega_c = zero_phonon;					// CAVITY FREQUENCY AT RESONANCE




//********************************************************************************************* PARAMETERS FOR SQUARE PULSE ********************************************************

double omega_d = 0.2;						// FREQUENCY OF SQUARE PULSE FOR INITIALIZATION
//double coup_d =  0.0;					        // AMPLITUDE OF THE PULSE
double coup_d = 0.0*3;					// AMPLITUDE OF THE PULSE
double t_stop = 0.0*15; 					// LENGTH OF SQUARE PULSE in us
//1.943*1.0E-7			

//********************************************************************************************* PARAMETERS FOR TIME EVOLUTION *****************************************************

double t_max = atof(argv[7]);					// T_END
int t_num = atoi(argv[8]);						// NUMBER OF STEPS
double t_step = t_max/t_num;					// dT (SIZE OF EACH STEP)



//********************************************************************************************* PARAMETERS FOR OUTPUT POINTS *****************************************************

int t_store_num = 20000;
int t_store =  t_num/t_store_num;

//***************************************************************************************** INITIALIZATION & PROGRAM BEGINS HERE **************************************************



// double N_a[num_ens],omega_a[num_ens],gamma_a[num_ens],\
		eta_a[num_ens],chi_a[num_ens],coup_a[num_ens],loss_a[num_ens];
double* N_a = new double[num_ens];
double* omega_a = new double[num_ens];
double* gamma_a = new double[num_ens];
double* eta_a = new double[num_ens];
double* chi_a = new double[num_ens];
double* coup_a = new double[num_ens];
double* loss_a = new double[num_ens];
 // SGK

for (int i =0; i < num_ens; i++){
	N_a[i] = ens_size;
	omega_a[i] = inhomo[i];
	gamma_a[i] = gamma_a_0;
	eta_a[i] = eta_a_0;
	chi_a[i] = chi_a_0;
	coup_a[i] = coup_a_0;
	loss_a[i] = loss_0;
}


// the parameters in an array 
// double para_a[7*num_ens];
// SGK
double* para_a = new double[7*num_ens];

for  (int i = 0; i < num_ens; i++){
	para_a[i] = N_a[i];
	para_a[i+num_ens] = omega_a[i];
	para_a[i+2*num_ens] = gamma_a[i];
	para_a[i+3*num_ens] = eta_a[i];
	para_a[i+4*num_ens] = chi_a[i];
	para_a[i+5*num_ens] = coup_a[i];
	para_a[i+6*num_ens] = loss_a[i];
}


// copy the parameters into the memory in GPU
double *para_a_dev;
cudaMalloc((void**)&para_a_dev,6*num_ens*sizeof(double)); 
cudaMemcpy(para_a_dev,para_a,6*num_ens*sizeof(double),cudaMemcpyHostToDevice);

//*******************************
// parameters for initial states 


// double theta[num_ens],phi[num_ens];
// SGK
double* theta = new double[num_ens];
double* phi = new double[num_ens];

for (int i=0; i < num_ens; i++){
	theta[i] = theta_0;
	phi[i] = phi_0;
}

// double2 cu[num_ens],cl[num_ens];
// SGK
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

//****************************
// initialize the observables
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

FILE *Result_time,*Result_Sz,*Result_Sm,*Result_photon,*Result_field, *coherences_real, *coherences_imag;
// time of simulation
Result_time = fopen("Result_time.dat","w");
Result_Sz = fopen("Result_Sz.dat","w");
Result_photon = fopen("Result_photon.dat","w");
Result_field = fopen("Result_field.dat","w");
Result_Sm = fopen("Result_Sm.dat","w");
coherences_real = fopen("coherences_real.dat","w");







// ***********************************
// simulations starts
// ***********************************
double tc;


// update the old reduced density matrix 
for (int t = 1; t < t_num; t++){

	tc = t*t_step;
	//printf("%1f \n", tc);

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
		fprintf(Result_field,"%e %e \n",a.x,a.y);
		//printf("%1f	%e	\n",tc, ap_a.x);
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		for (int i = 0; i < num_ens; i++) {
			fprintf(Result_Sz,"%e ",sz[i].x);
			fprintf(Result_Sm,"%e %e  ",sm[i].x, sm[i].y);
			fprintf(coherences_real,"%e ",sm_sp[i].x);
		}
		fprintf(Result_Sz,"\n");
		fprintf(Result_Sm,"\n");
		fprintf(coherences_real,"\n");
		
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
}


// close the files
fclose(Result_time);
fclose(Result_Sz);
fclose(Result_photon);
fclose(Result_field);
fclose(coherences_real);


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



ct1=clock();
fprintf(stderr,"Program takes about %.2f s\n",(double)(ct1-ct0)/(double)CLOCKS_PER_SEC);
return 0;
}
