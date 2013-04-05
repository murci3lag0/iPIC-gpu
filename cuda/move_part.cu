
// update the particle positions
__global__ void move_part(int np, float dt, float Lx, float *x, float *u) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < np) {
        x[tid] = x[tid] + u[tid]*float(dt);

        if (x[tid] < float(0.0)) {
            x[tid] = x[tid] + float(Lx);
        }
        if (x[tid]>= float(Lx)) {
            x[tid] = x[tid] - float(Lx);
        }
        tid += blockDim.x * gridDim.x;
    }

}
