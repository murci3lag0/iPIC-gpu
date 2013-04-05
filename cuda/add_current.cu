#include "include/gputools.h"

// sum up the currents per cell for all particles
__global__ void add_current(int np, float wpi, float qom, float Lx, float dx, float *Jx, float *x, float *u) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < np) {

        float q = float(wpi)*float(wpi)/(float(qom)*float(np))/float(Lx);

        ATOMADD(&Jx[int(floorf(x[tid]/float(dx)))], float(q)*u[tid]/float(2.0));

        tid += blockDim.x * gridDim.x;
    }
}    
