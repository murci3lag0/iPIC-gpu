
// update the velocities
__global__ void update_vel(int npart, float qom, float dx, float dt, float *Epx, float *u){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while ( tid < npart) {

        u[tid] = u[tid] + float(qom)*Epx[tid]*float(dt);

        tid += blockDim.x * gridDim.x;
    }
}
