#include "include/numerics.h"

Numerics::Numerics(Collective *col){

    acc    = col->get_acc();
    iorder = col->get_iorder();
    dt     = col->dt;
}

void Numerics::CalculateFields(Grid *grid, Fields *fields){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
            CalculateFields_gpu(grid, fields);
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
            CalculateFields_mic(grid, fields);
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Numerics::CalculateFields_gpu(Grid *grid, Fields *fields){
#ifdef __GPU__
    if (grid->ncx > 1) calc_field_gpu(nblock_fields, thperbloc_fields, grid->get_ncell(), float(dt), (float*)fields->dev_Jx, (float*)fields->dev_Ex);
    if (grid->ncy > 1) calc_field_gpu(nblock_fields, thperbloc_fields, grid->get_ncell(), float(dt), (float*)fields->dev_Jy, (float*)fields->dev_Ey);
    if (grid->ncz > 1) calc_field_gpu(nblock_fields, thperbloc_fields, grid->get_ncell(), float(dt), (float*)fields->dev_Jz, (float*)fields->dev_Ez);
#endif
}

void Numerics::CalculateFields_mic(Grid *grid, Fields *fields){
#ifdef __MIC__
#endif
}

void Numerics::MoveParticles(Grid *grid, Particles *part){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
            MoveParticles_gpu(grid, part);
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
            MoveParticles_mic(grid, part);
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Numerics::MoveParticles_gpu(Grid *grid, Particles *part){
#ifdef __GPU__
    if (grid->ncx > 1) move_part_gpu(nblock_part, thperbloc_part, part->npart, float(dt), float(grid->get_Lx()), (float*)part->dev_x, (float*)part->dev_u);
    if (grid->ncy > 1) move_part_gpu(nblock_part, thperbloc_part, part->npart, float(dt), float(grid->get_Ly()), (float*)part->dev_y, (float*)part->dev_v);
    if (grid->ncz > 1) move_part_gpu(nblock_part, thperbloc_part, part->npart, float(dt), float(grid->get_Lz()), (float*)part->dev_z, (float*)part->dev_w);
#endif
}

void Numerics::MoveParticles_mic(Grid *grid, Particles *part){
#ifdef __MIC__
    move_part_mic(nblock_part, thperbloc_part, part->npart, (float*)part->dev_x, (float*)part->dev_u);
#endif
}

void Numerics::UpdateFields(Grid *grid, Fields *fields){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
            UpdateFields_gpu(grid, fields);
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
            UpdateFields_mic(grid, fields);
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Numerics::UpdateFields_gpu(Grid *grid, Fields *fields){
#ifdef __GPU__
    set_current_gpu(nblock_fields, thperbloc_fields, grid->get_ncell(), 
                    (float*)fields->dev_Jx,
                    (float*)fields->dev_Jy,
                    (float*)fields->dev_Jz);
#endif
}

void Numerics::UpdateFields_mic(Grid *grid, Fields *fields){
}

void Numerics::UpdateVelocity(Grid *grid, Particles *part, Plasma *plasma){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
            UpdateVelocity_gpu(grid, part, plasma);
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
            UpdateVelocity_mic(grid, part, plasma);
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Numerics::UpdateVelocity_gpu(Grid *grid, Particles *part, Plasma *plasma){
#ifdef __GPU__
    if (grid->ncx > 1) update_vel_gpu(nblock_part, thperbloc_part, part->npart, plasma->qom, grid->get_dx(), float(dt), (float*)part->dev_Ex, (float*)part->dev_u);
    if (grid->ncy > 1) update_vel_gpu(nblock_part, thperbloc_part, part->npart, plasma->qom, grid->get_dy(), float(dt), (float*)part->dev_Ey, (float*)part->dev_v);
    if (grid->ncz > 1) update_vel_gpu(nblock_part, thperbloc_part, part->npart, plasma->qom, grid->get_dz(), float(dt), (float*)part->dev_Ez, (float*)part->dev_w);
#endif
}

void Numerics::UpdateVelocity_mic(Grid *grid, Particles *part, Plasma *plasma){
}

void Numerics::InitHardware(Grid *grid, Fields *fields, Particles *part){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
#ifdef __GPU__
            Init_gpu(grid, fields, part);
#endif
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
#ifdef __MIC__
            Init_mic(grid, fields, part);
#endif
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Numerics::StopHardware(Fields *fields, Particles *part){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
#ifdef __GPU__
            Stop_gpu(fields, part);
#endif
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
#ifdef __MIC__
            Stop_mic();
#endif
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

Numerics::~Numerics(){
}
