
// set the current Jx on each cell
__global__ void set_current(int ncells, float *Jx, float *Jy, float *Jz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < ncells) {
        Jx[tid] = Jx[tid]/float(2.0);
        Jy[tid] = Jy[tid]/float(2.0);
        Jz[tid] = Jz[tid]/float(2.0);

        tid += blockDim.x * gridDim.x;
    }
}
