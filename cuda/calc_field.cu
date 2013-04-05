
// calculate the field in each cell
__global__ void calc_field(int ncells, float dt, float *Jx, float *Ecx){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid<ncells) {
        Ecx[tid] = Ecx[tid] - Jx[tid]*float(dt);

        tid += blockDim.x * gridDim.x;
    }
}
