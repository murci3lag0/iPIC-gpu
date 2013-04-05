#include "include/fields.h"

Fields::Fields(Collective *col){

    int nx = col->ncx;
    int ny = col->ncy;
    int nz = col->ncz;

    allocatefields(nx,ny,nz);
}

void Fields::allocatefields(int nx, int ny, int nz){

    // Allocate J
    Jx = new float[nx*ny*nz];
    Jy = new float[nx*ny*nz];
    Jz = new float[nx*ny*nz];

    // Allocate E
    Ex = new float[nx*ny*nz];
    Ey = new float[nx*ny*nz];
    Ez = new float[nx*ny*nz];

}

void Fields::InitFieldsZero(Grid *grid){

    for (int i = 0; i < grid->get_ncell(); i++) {
        Jx[i]     = 0.0;
        Jy[i]     = 0.0;
        Jz[i]     = 0.0;

        Ex[i]     = 0.0;
        Ey[i]     = 0.0;
        Ez[i]     = 0.0;
    }
}

Fields::~Fields(){
}
