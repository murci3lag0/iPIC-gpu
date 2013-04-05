#include "include/grid.h"

Grid::Grid(Collective *col){

    // Geometry
    Lx = col->Lx;
    Ly = col->Ly;
    Lz = col->Lz;

    ncx   = col->ncx;
    ncy   = col->ncy;
    ncz   = col->ncz;

    ncell = ncx * ncy * ncz;

    dx = Lx / float(ncx);
    dy = Ly / float(ncy);
    dz = Lz / float(ncz);

    // Boundary conditions

}

int Grid::get_ncell(){
    return ncell;
}

float Grid::get_Lx(){
    return Lx;
}

float Grid::get_Ly(){
    return Ly;
}

float Grid::get_Lz(){
    return Lz;
}

float Grid::get_dx(){
    return dx;
}

float Grid::get_dy(){
    return dy;
}

float Grid::get_dz(){
    return dz;
}

Grid::~Grid(){
}

