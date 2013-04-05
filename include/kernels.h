#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "include/gputools.h"

#ifdef __GPU__
__global__ void set_current(int ncells, float *Jx, float *Jy, float *Jz);
__global__ void move_part(int np, float dt, float Lx, float *x, float *u);
__global__ void add_current(int np, float wpi, float qom, float Lx, float dx, float *Jx, float *x, float *u);
__global__ void calc_field(int ncell, float dt, float *Jx, float *Ecx);
__global__ void update_vel(int npart, float qom, float dx, float dt, float *Epx, float *u);
__global__ void fields2part(int np, float dx, float *x, float *Epx, float *Ecx);
#endif

#endif
