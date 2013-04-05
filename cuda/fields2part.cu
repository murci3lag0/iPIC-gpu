
// update the velocities
__global__ void fields2part(int np, float dx, float *xp, float *Epx, float *Ecx){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < np) {

        Epx[tid] = Ecx[int(xp[tid]/float(dx))];

        tid += blockDim.x * gridDim.x;
    }
}
