#ifndef __NUMERICS_GPU_CUH__
#define __NUMERICS_GPU_CUH__

#include "include/grid.h"
#include "include/fields.h"
#include "include/particles.h"

#ifdef __GPU__
#include "include/kernels.h"
#endif

// These values must be calculated and not hardcoded
// ---   WARNING: change the next four lines
// ---   WARNING: change this values also in numerics and interpolation classes
#define nblock_fields     64
#define thperbloc_fields  4
#define nblock_part       32768
#define thperbloc_part    4
// ---   END WARNING

void Init_gpu(Grid *grid, Fields *fields, Particles *part);
void Stop_gpu(Fields *fields, Particles *part);
void MemCpy_cpu2gpu(Grid *grid, Fields *fields, Particles *part);
void MemCpy_gpu2cpu(Grid *grid, Fields *fields, Particles *part);

// Numerics on GPU:
void set_current_gpu(int nb_fields, int tpb_fields, int ncell, float *Jx, float *Jy, float *Jz);
void move_part_gpu(int nb_part, int tpb_part, int npart, float dt, float Lx, float *x, float *u);
void calc_field_gpu(int nb_fields, int tpb_fields, int ncell, float dt, float *Jx, float *Ex);
void update_vel_gpu(int nb_part, int tpb_part, int npart, float qom, float dx, float dt, float *Epx, float *u);

// Interpolation on GPU:
void add_current_gpu(int nb_part, int tpb_part, int npart, float wpi, float qm, float Lx, float dx, float *Jx, float *x, float *u);
void fields2part_gpu(int nb_part, int tpb_part, int npart, float dx, float *xp, float *Epx, float *Ecx);

#endif
