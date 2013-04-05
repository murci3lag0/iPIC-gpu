#include "include/interpolation.h"

Interpolation::Interpolation(Collective *col){

    // Select the interpolation method
    interptype = col->get_interptype();
    // Set the type of accelerator used in the interpolation
    acc = col->get_acc();
}

void Interpolation::Part2Fields_cpu2cpu(Grid *grid, Fields *fields, Particles *part, Plasma *plasma){
}

void Interpolation::Part2Fields(Grid *grid, Fields *fields, Particles *part, Plasma *plasma){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
            Part2Fields_gpu(grid, fields, part, plasma);
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
            Part2Fields_mic(grid, fields, part, plasma);
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Interpolation::Part2Fields_gpu(Grid *grid, Fields *fields, Particles *part, Plasma *plasma){
#ifdef __GPU__
    if (grid->ncx > 1) add_current_gpu(nblock_part, thperbloc_part, part->npart, plasma->wpi, plasma->qom,
                                       grid->get_Lx(), grid->get_dx(),
                                       (float*)fields->dev_Jx, (float*)part->dev_x, (float*)part->dev_u);

    if (grid->ncy > 1) add_current_gpu(nblock_part, thperbloc_part, part->npart, plasma->wpi, plasma->qom,
                                       grid->get_Ly(), grid->get_dy(),
                                       (float*)fields->dev_Jy, (float*)part->dev_y, (float*)part->dev_v);

    if (grid->ncz > 1) add_current_gpu(nblock_part, thperbloc_part, part->npart, plasma->wpi, plasma->qom,
                                       grid->get_Lz(), grid->get_dz(),
                                       (float*)fields->dev_Jz, (float*)part->dev_z, (float*)part->dev_w);
#endif
}

void Interpolation::Part2Fields_mic(Grid *grid, Fields *fields, Particles *part, Plasma *plasma){
#ifdef __MIC__
#endif
}

void Interpolation::Fields2Part(Grid *grid, Fields *fields, Particles *part){
    switch(acc){
        case(0) :
            break;
        case(1) : // GPU
            Fields2Part_gpu(grid, fields, part);
            break;
        case(2) : // MPI
            break;
        case(3) : // OpenMP
            break;
        case(4) : // MIC
            Fields2Part_mic(grid, fields, part);
            break;
        default:
            cout << " ERROR: Unrecognized accelerator = " << acc << endl;
            abort();
            break;
    }
}

void Interpolation::Fields2Part_gpu(Grid *grid, Fields *fields, Particles *part){
#ifdef __GPU__
    if (grid->ncx > 1) fields2part_gpu(nblock_part, thperbloc_part, part->npart, grid->get_dx(), (float*)part->dev_x, (float*)part->dev_Ex, (float*)fields->dev_Ex);
    if (grid->ncy > 1) fields2part_gpu(nblock_part, thperbloc_part, part->npart, grid->get_dy(), (float*)part->dev_y, (float*)part->dev_Ey, (float*)fields->dev_Ey);
    if (grid->ncz > 1) fields2part_gpu(nblock_part, thperbloc_part, part->npart, grid->get_dz(), (float*)part->dev_z, (float*)part->dev_Ez, (float*)fields->dev_Ez);
#endif
}

void Interpolation::Fields2Part_mic(Grid *grid, Fields *fields, Particles *part){
#ifdef __MIC__
#endif
}

Interpolation::~Interpolation(){
}
