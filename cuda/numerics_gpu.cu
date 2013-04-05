#include "include/numerics_gpu.h"

void set_current_gpu(int nb_fields, int tpb_fields, int ncell, float *Jx, float *Jy, float *Jz){
    set_current<<<nb_fields,tpb_fields>>>(ncell, Jx, Jy, Jz);
}

void move_part_gpu(int nb_part, int tpb_part, int npart, float dt, float Lx, float *x, float *u){
    move_part<<<nb_part,tpb_part>>>(npart, dt, Lx, x, u);
}

void add_current_gpu(int nb_part, int tpb_part, int npart, float wpi, float qm, float Lx, float dx, float *Jx, float *x, float *u){
    add_current<<<nb_part,tpb_part>>>(npart, wpi, qm, Lx, dx, Jx, x, u);
}

void calc_field_gpu(int nb_fields, int tpb_fields, int ncell, float dt, float *Jx, float *Ex){
    calc_field<<<nb_fields,tpb_fields>>>(ncell, dt, Jx, Ex);
}

void fields2part_gpu(int nb_part, int tpb_part, int npart, float dx, float *xp, float *Epx, float *Ecx){
    fields2part<<<nb_part,tpb_part>>>(npart, dx, xp, Epx, Ecx);
}

void update_vel_gpu(int nb_part, int tpb_part, int npart, float qom, float dx, float dt, float *Epx, float *u){
    update_vel<<<nb_part,tpb_part>>>(npart, qom, dx, dt, Epx, u);
}

void Init_gpu(Grid *grid, Fields *fields, Particles *part){
    HANDLE_ERROR( cudaMalloc( (void**)&fields->dev_Jx, grid->get_ncell()*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&fields->dev_Jy, grid->get_ncell()*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&fields->dev_Jz, grid->get_ncell()*sizeof(float)) );

    HANDLE_ERROR( cudaMalloc( (void**)&fields->dev_Ex, grid->get_ncell()*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&fields->dev_Ey, grid->get_ncell()*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&fields->dev_Ez, grid->get_ncell()*sizeof(float)) );

    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_x, part->npart*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_y, part->npart*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_z, part->npart*sizeof(float)) );

    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_u, part->npart*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_v, part->npart*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_w, part->npart*sizeof(float)) );

    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_Ex, part->npart*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_Ey, part->npart*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&part->dev_Ez, part->npart*sizeof(float)) );
}

void Stop_gpu(Fields *fields, Particles *part){
    HANDLE_ERROR( cudaFree( fields->dev_Jx ) );
    HANDLE_ERROR( cudaFree( fields->dev_Jy ) );
    HANDLE_ERROR( cudaFree( fields->dev_Jz ) );

    HANDLE_ERROR( cudaFree( fields->dev_Ex ) );
    HANDLE_ERROR( cudaFree( fields->dev_Ey ) );
    HANDLE_ERROR( cudaFree( fields->dev_Ez ) );

    HANDLE_ERROR( cudaFree( part->dev_x ) );
    HANDLE_ERROR( cudaFree( part->dev_y ) );
    HANDLE_ERROR( cudaFree( part->dev_z ) );

    HANDLE_ERROR( cudaFree( part->dev_u ) );
    HANDLE_ERROR( cudaFree( part->dev_v ) );
    HANDLE_ERROR( cudaFree( part->dev_w ) );

    HANDLE_ERROR( cudaFree( part->dev_Ex ) );
    HANDLE_ERROR( cudaFree( part->dev_Ey ) );
    HANDLE_ERROR( cudaFree( part->dev_Ez ) );
}

void MemCpy_cpu2gpu(Grid *grid, Fields *fields, Particles *part){
    HANDLE_ERROR( cudaMemcpy( fields->dev_Jx, fields->Jx, grid->get_ncell()*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( fields->dev_Jy, fields->Jy, grid->get_ncell()*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( fields->dev_Jz, fields->Jz, grid->get_ncell()*sizeof(float), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( fields->dev_Ex, fields->Ex, grid->get_ncell()*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( fields->dev_Ey, fields->Ey, grid->get_ncell()*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( fields->dev_Ez, fields->Ez, grid->get_ncell()*sizeof(float), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( part->dev_x, part->x, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( part->dev_y, part->y, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( part->dev_z, part->z, part->npart*sizeof(float), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( part->dev_u, part->u, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( part->dev_v, part->v, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( part->dev_w, part->w, part->npart*sizeof(float), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( part->dev_Ex, part->Ex, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( part->dev_Ey, part->Ey, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( part->dev_Ez, part->Ez, part->npart*sizeof(float), cudaMemcpyHostToDevice) );
}

void MemCpy_gpu2cpu(Grid *grid, Fields *fields, Particles *part){

    HANDLE_ERROR( cudaMemcpy( fields->Jx, fields->dev_Jx, grid->get_ncell()*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( fields->Jy, fields->dev_Jy, grid->get_ncell()*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( fields->Jz, fields->dev_Jz, grid->get_ncell()*sizeof(float), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy( fields->Ex, fields->dev_Ex, grid->get_ncell()*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( fields->Ey, fields->dev_Ey, grid->get_ncell()*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( fields->Ez, fields->dev_Ez, grid->get_ncell()*sizeof(float), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy( part->x, part->dev_x, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( part->y, part->dev_y, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( part->z, part->dev_z, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy( part->u, part->dev_u, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( part->v, part->dev_v, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( part->w, part->dev_w, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy( part->Ex, part->dev_Ex, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( part->Ey, part->dev_Ey, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( part->Ez, part->dev_Ez, part->npart*sizeof(float), cudaMemcpyDeviceToHost) );

}
