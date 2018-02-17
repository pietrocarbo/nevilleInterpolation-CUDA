#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <string.h>
#include <time.h>

#define MAX_BLOCKS_X 65535 	// # max di blocchi sulla dimensione .x della griglia per archittetura Fermi
#define N 32				// punti di interpolazione



/**		SERIALE
*  px , py vettori di 32 elementi nodi interpolazione
*  x vettore di Nx elementi nei quali valutare P(x_i)
*  y vettore di Nx elementi in cui memorizzare le valutazioni
*/
void serialNevilleInterpolation (double *x,  double *y, int Nx, double *px, double *py) {
 	double s[32], xe;
 	int i, ii, jj;

 	// itero su tutti i punti di valutazione
	for ( i = 0; i < Nx; i++) {		
		xe = x[i];

		// preparo vettore s temporaneo al calcolo
		for (ii = 0; ii < 32; ii++) 
			s[ii] = py[ii];
		
		
		for ( ii = 1; ii <= 31; ii++) {
			for ( jj = 0; jj <= 31 - ii; jj++) {
				// IMPLEMENTAZIONE ALGORITMO NEVILLE
				// x_i 			=>  i 			= jj
				// x_j  		=>  j 			= jj + ii
				// P_(i, j-1)	=>  (i, j-1) 	= jj
				// P_(i+1, j)  	=>  (i+1, j)	= jj + 1

				s[jj] = ((( px[jj + ii] - xe ) * s[jj] ) + ((xe - px[jj] ) * s[jj+1] )) / ( px[jj + ii] - px[jj] );
			}
		}
		y[i] = s[0];
	}
}



/**		KERNEL 0
*  px , py vettori di 32 elementi nodi interpolazione
*  x vettore di Nx elementi nei quali valutare P(x_i)
*  y vettore di Nx elementi in cui memorizzare le valutazioni
*/
__global__ void kernel_0 (double *x, double *y, int Nx, double *px, double *py) {

	int blockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if (blockIndex >= Nx)	return;

	int ii, jj, threadId = threadIdx.x;
	double xe = x[blockIndex];

	__shared__ double spx[N];
	__shared__ double spy[N];
	
	spy[threadId] = py[threadId];
	spx[threadId] = px[threadId];

	if (threadId == 0) {
		for (ii = 1; ii <= 31; ii++ ) {
			for (jj = 0; jj <= 31 - ii; jj++ )
				spy[jj] = ( (spx[jj + ii] - xe ) * spy[jj]  + (xe - spx[jj] ) * spy[jj+1] ) / ( spx[jj + ii] - spx[jj] );
		}
		y[blockIndex] = spy[0];
	}
}
 


/**		KERNEL 1
*  px , py vettori di 32 elementi nodi interpolazione
*  x vettore di Nx elementi nei quali valutare P(x_i)
*  y vettore di Nx elementi in cui memorizzare le valutazioni
*/
__global__ void kernel_1 (double *x, double *y, int Nx, double *px, double *py) {
	
	int blockIndex = blockIdx.y * gridDim.x + blockIdx.x;
	if (blockIndex >= Nx)	return;

	int ii, j, threadId = threadIdx.x;
	double xe = x[blockIndex], s0_tmp, s1_tmp;

	__shared__ double spx[N];
	__shared__ double spy[N];
	
	spy[threadId] = py[threadId];
	spx[threadId] = px[threadId];
	 
	for (ii = 1; ii <= 31; ii++) {
		j = (ii + threadId) % N;
		s0_tmp = spy[threadId];
		s1_tmp = spy[(threadId + 1) % N];

		// __syncthreads();

		spy[threadId] = ( (spx[j] - xe) * s0_tmp  + (xe - spx[threadId]) * s1_tmp ) / (spx[j] - spx[threadId]);
	}
	if(threadId == 0)	y[blockIndex] = spy[0];
} 


/* Funzione di utility che verifica se una stringa rappresenta un intero
 */
bool isInteger (char number[]) {
    int i = 0;
    if  (number[0] == '-')	i = 1;  else if (number[0] == 0)	return false;
    for ( ;number[i] != 0; i++) 	if (!isdigit(number[i]))	return false;
    return true;
}

/*	Funzione che esegue il parsing degli argomenti dati al programma da riga di comando
 */
int examine_args ( int argc, char* argv[], unsigned int* Nx ) {
	int i, exec_mode = 0;
	*Nx = 200000;

	if (argc == 1)		return exec_mode;

	char helper[] = "[!] Usage of nevilleInterpolation [!]\n"
	"\t ./nevilleInterpolation [-s] [-k0] [-k1] [-nx number] \n"
	"\t -s  \t To run only the serial implementation.\n"
	"\t -k0 \t To run only the kernel_0.\n"
	"\t -k1 \t To run only the kernel_1.\n"
	"\t -nx <number> \t To specify an exact number of random points on which we will interpolate on. <number> MUST be greater than zero.\n"
	"Without any options the DEFAULT values will be the execution of all procedures on [Nx] 200000 points.";

	if (argc > 4) {
		printf("%s\n", helper);
		return -1;
	}

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-nx") == 0) {
			if (argc <= i+1 || !isInteger(argv[i+1]) || atoi(argv[i+1]) <= 0) {
				printf("%s\n", helper);
				return -1;
			} 
			*Nx =  atoi(argv[i+1]);
			i++;
		} 
		else if (strcmp(argv[i], "-s") == 0) 		exec_mode = 1;
		else if (strcmp(argv[i], "-k0") == 0) 		exec_mode = 2;
		else if (strcmp(argv[i], "-k1") == 0)		exec_mode = 3;
		else if (strcmp(argv[i], "--help") == 0) {
			printf("%s\n", helper);
			return -1;
		} 
		else {
			printf("%s\n", helper);
			return -1;
		}	
	}
	return exec_mode;
}



int main (int argc, char* argv[]) {

	unsigned int dGx, dGy, ii, errors = 0, Nx;
	int exec_mode = examine_args(argc, argv, &Nx);
	if (exec_mode == -1)	return 1;

	// Nomenclatura variabili contenti i dati-> [px/py] nodi, [x/y] punti da interpolare (d:Device, h:Host)
	double *pxh, *pyh, *pxd, *pyd, *xh, *yh, *xd, *yd;
	double PI = 4 * atan(1.0), step = 1.0 / (N - 1), fx, fx_appr, xi;

	// Variabili per ottenere i tempi di esecuzione
	float elapsedTimeCpu, elapsedTimeGpu, elapsedTimeDataH2D, elapsedTimeDataD2H, totalTimeGPU;
	cudaEvent_t startGpu, stopGpu;
	clock_t startCpu, stopCpu;
	
	// Dimensionamento blocco e griglia di threads
	dim3 dimensioniBlocco (N, 1, 1);
	if (Nx > MAX_BLOCKS_X) {
		dGx = MAX_BLOCKS_X;
		dGy = (Nx + MAX_BLOCKS_X - 1) / MAX_BLOCKS_X;
	} else {
		dGx = Nx;
		dGy = 1;
	}
	dim3 dimensioniGriglia (dGx, dGy, 1);
	fprintf(stderr, "Dimensioni problema: Nx = %d   Griglia (%d, %d, 1) \n", Nx, dGx, dGy);

	// Alloco spazio in memoria host
	xh  = (double*) malloc(Nx * sizeof(double));
	yh  = (double*) malloc(Nx * sizeof(double));
	pxh = (double*) malloc(N  * sizeof(double));
	pyh = (double*) malloc(N  * sizeof(double));

	srand(123);
	for (ii = 0; ii < Nx; ii++)   // random points in [0,PI]
		xh[ii] = PI * rand() / (double) RAND_MAX; 

	for (ii = 0; ii < N; ii++) {
		pxh[ii] = ii * step * PI;  //punti equispaziati  
		pyh[ii] = sin(pxh[ii]);    
	}


	// ESECUZIONE CODICE SERIALE
	if (exec_mode == 0 || exec_mode == 1) {

		startCpu = clock();
		serialNevilleInterpolation(xh, yh, Nx, pxh, pyh);
		stopCpu = clock();

		elapsedTimeCpu = ((float) (stopCpu - startCpu) * 1000.0) / CLOCKS_PER_SEC;

		for (ii = 0; ii < Nx; ii++) {
			xi = xh[ii];
			fx = sin(xi);
			fx_appr = yh[ii];

			#ifdef POINTS
				printf("%d)\t sin(%f) = %lf ~> %lf \n", ii+1, xi, fx, fx_appr);
			#endif	
			
			errors = fabs(fx_appr - fx) / fabs(fx);
			if ( errors > 1e-10)		errors++;
		}	// Stampa delle metriche
		fprintf(stderr, "Seriale==>  T:%8.3f[ms]  E:%d\n", elapsedTimeCpu, errors);

	}


	// ESECUZIONE CODICE PARALLELO CUDA
	if (exec_mode == 0 || exec_mode == 2 || exec_mode == 3) {
		
		// Allocazione e copia dei vettori nella memoria globale GPU
		cudaEventCreate(&startGpu);
		cudaEventCreate(&stopGpu);
		cudaEventRecord(startGpu, 0);
		checkCudaErrors ( cudaMalloc ( (void **) &pxd, 	N  * sizeof(double)) );
		checkCudaErrors ( cudaMalloc ( (void **) &pyd, 	N  * sizeof(double)) );
		checkCudaErrors ( cudaMalloc ( (void **) &xd, 	Nx * sizeof(double)) );
		checkCudaErrors ( cudaMalloc ( (void **) &yd, 	Nx * sizeof(double)) );
		checkCudaErrors ( cudaMemcpy ( pxd, pxh, N  *  sizeof(double), cudaMemcpyHostToDevice) );
		checkCudaErrors ( cudaMemcpy ( pyd, pyh, N  *  sizeof(double), cudaMemcpyHostToDevice) );
		checkCudaErrors ( cudaMemcpy ( xd,  xh,  Nx *  sizeof(double), cudaMemcpyHostToDevice) );
		cudaEventRecord(stopGpu, 0);
		cudaEventSynchronize( stopGpu ); 
		cudaEventElapsedTime( &elapsedTimeDataH2D, startGpu, stopGpu );   // tempo di trasferimento Host to Device

		// LANCIO KERNEL_0
		if (exec_mode == 0 || exec_mode == 2) {			
			cudaEventRecord(startGpu, 0);
			kernel_0 <<< dimensioniGriglia, dimensioniBlocco >>> (xd, yd, Nx, pxd, pyd);
			cudaEventRecord(stopGpu, 0);
			cudaEventSynchronize( stopGpu ); 
			cudaEventElapsedTime( &elapsedTimeGpu, startGpu, stopGpu ); 
			
			cudaEventRecord(startGpu, 0);
			checkCudaErrors ( cudaMemcpy (yh, yd, Nx * sizeof(double), cudaMemcpyDeviceToHost) );
			cudaEventRecord(stopGpu, 0);
			cudaEventSynchronize( stopGpu ); 
			cudaEventElapsedTime( &elapsedTimeDataD2H, startGpu, stopGpu );	// tempo di trasferimento Device to Host
			totalTimeGPU = elapsedTimeGpu + elapsedTimeDataD2H + elapsedTimeDataH2D;			

			errors = 0;  // Conteggio errori
			for (ii = 0; ii < Nx; ii++) {
				xi = xh[ii];
				fx = sin(xi);
				fx_appr = yh[ii];

				#ifdef POINTS
					printf("%d)\t sin(%f) = %lf ~> %lf \n", ii+1, xi, fx, fx_appr);
				#endif	
				
				errors = fabs(fx_appr - fx) / fabs(fx);
				if ( errors > 1e-10)		errors++;
			}	
			fprintf(stderr, "Kernel0==>  T:%8.3f[ms]  TD:%8.3f[ms]  S:%6.6f  SD:%6.6f  BP:%5.8f[GB/s]  E:%d\n", 
			elapsedTimeGpu, totalTimeGPU, elapsedTimeCpu/elapsedTimeGpu, elapsedTimeCpu/totalTimeGPU,
			(sizeof(double) * Nx * 1e-9) / (elapsedTimeGpu * 1e-3), errors);
		}

		// LANCIO KERNEL_1
		if (exec_mode == 0 || exec_mode == 3) {
			cudaEventRecord(startGpu, 0);
			kernel_1 <<< dimensioniGriglia, dimensioniBlocco >>> (xd, yd, Nx, pxd, pyd);
			cudaEventRecord(stopGpu, 0);
			cudaEventSynchronize( stopGpu ); 
			cudaEventElapsedTime( &elapsedTimeGpu, startGpu, stopGpu ); 
			
			cudaEventRecord(startGpu, 0);
			checkCudaErrors ( cudaMemcpy (yh, yd, Nx * sizeof(double), cudaMemcpyDeviceToHost) );
			cudaEventRecord(stopGpu, 0);
			cudaEventSynchronize( stopGpu ); 
			cudaEventElapsedTime( &elapsedTimeDataD2H, startGpu, stopGpu );	
			totalTimeGPU = elapsedTimeGpu + elapsedTimeDataD2H + elapsedTimeDataH2D;			

			errors = 0;
			for (ii = 0; ii < Nx; ii++) {
				xi = xh[ii];
				fx = sin(xi);
				fx_appr = yh[ii];

				#ifdef POINTS
					printf("%d)\t sin(%f) = %lf ~> %lf \n", ii+1, xi, fx, fx_appr);
				#endif	
				
				errors = fabs(fx_appr - fx) / fabs(fx);
				if ( errors > 1e-10)		errors++;
			}
			fprintf(stderr, "Kernel1==>  T:%8.3f[ms]  TD:%8.3f[ms]  S:%6.6f  SD:%6.6f  BP:%5.8f[GB/s]  E:%d\n", 
			elapsedTimeGpu, totalTimeGPU, elapsedTimeCpu/elapsedTimeGpu, elapsedTimeCpu/totalTimeGPU,
			(sizeof(double) * Nx * 1e-9) / (elapsedTimeGpu * 1e-3), errors);
		}
		cudaEventDestroy(startGpu);
		cudaEventDestroy(stopGpu);
		cudaFree(pxd);
		cudaFree(pyd);
		cudaFree(yd);
		cudaFree(xd);
	}
	free(pxh);	
	free(pyh);	
	free(xh);	
	free(yh);	 
	return 1;
}
