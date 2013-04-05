#include "include/particles.h"

Particles::Particles(Collective *col, Grid *grid){

    npart = col->npart;
    ppc   = grid->get_ncell() / npart;

    AllocateParticles();
}

void Particles::AllocateParticles(){

    x = new float[npart];
    y = new float[npart];
    z = new float[npart];

    u = new float[npart];
    v = new float[npart];
    w = new float[npart];

    Ex = new float[npart];
    Ey = new float[npart];
    Ez = new float[npart];

}

void Particles::GenerateParticles(Collective *col, Grid *grid){

    // Init the interpolated fields->particles to zero
    PartFieldsZero();

    // Spatial distribution
    switch (col->get_xdistri()) {
        case 1:  // Random
            distx_random(grid);
            break;
        default:
            break;
    }

    // Velocity distribution
    switch (col->get_udistri()) {
        case 1: // Drift + random thermal
            distu_randomthermal(col,grid);
            break;
        case 2: // Maxwellian
            distu_maxwellian(col,grid);
            break;
        default:
            break;
    }

}

void Particles::PartFieldsZero(){
    for (int i = 0; i < npart; i++){
        Ex[i] = 0.0;
        Ey[i] = 0.0;
        Ez[i] = 0.0;
    }
}

void Particles::distx_random(Grid *grid){

    for (int i = 0; i < npart; i++){
        x[i] = grid->ncx <= 1 ? 0 : ((float)rand() / (float)RAND_MAX) * grid->get_Lx();
        y[i] = grid->ncy <= 1 ? 0 : ((float)rand() / (float)RAND_MAX) * grid->get_Ly();
        z[i] = grid->ncz <= 1 ? 0 : ((float)rand() / (float)RAND_MAX) * grid->get_Lz();
    }

}

void Particles::distu_randomthermal(Collective *col, Grid *grid){

    for (int i = 0; i < npart; i++){
        u[i] = grid->ncx <= 1 ? 0 : float(pow(float(-1.0),float(i)) * (float(col->u0) + float(col->ut)*((float)rand() / (float)RAND_MAX)));
        v[i] = grid->ncy <= 1 ? 0 : float(pow(float(-1.0),float(i)) * (float(col->v0) + float(col->vt)*((float)rand() / (float)RAND_MAX)));
        w[i] = grid->ncz <= 1 ? 0 : float(pow(float(-1.0),float(i)) * (float(col->w0) + float(col->wt)*((float)rand() / (float)RAND_MAX)));
    }

}

void Particles::distu_maxwellian(Collective *col, Grid *grid){

    for (int i = 0; i < npart; i++){
        u[i] = grid->ncx <= 1 ? 0 : 1;
        v[i] = grid->ncy <= 1 ? 0 : 1;
        w[i] = grid->ncz <= 1 ? 0 : 1;
    }

}

void Particles::AddPerturbation(Collective *col, Grid *grid){

    // Position and velocity perturbation
    switch(col->ipert) {
        case 1 :
            // Sinusoidal perturbation in the x direction
            for (int i = 0; i < npart; i++) {
                u[i] = u[i] + float(col->u_pert) * sin(float(2.0)*float(pi)*x[i] / float(grid->get_Lx()) * float(mode));
                x[i] = x[i] + float(col->x_pert) * sin(float(2.0)*float(pi)*x[i] / float(grid->get_Lx()) * float(mode)) * (float(grid->get_Lx())/float(npart));
            }
            break;
        default :
            break;
    }

    // Effects on the current field
    // THIS MUST GO IN THE INTERPOLATOR CLASS
    //
    //for (int i = 0; i < npart; i++) {

    //    int icell = floorf(xp[ip] / (Lx/ncells));
    //    float q   = float(wpi)*float(wpi)/(float(qm)*float(np))/float(Lx);

    //    float jx_update = q * up[ip];

    //    for (int ic = 0; ic < ncells; ic++) {
    //        if (ic == icell) {
    //            jx_old[ic] = jx_old[ic] + jx_update;
    //        }
    //    }

    //}

}

Particles::~Particles(){
}
