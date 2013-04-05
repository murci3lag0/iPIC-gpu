
#ifndef __GRID_H__
#define __GRID_H__

#include "include/collective.h"

class Grid {
    public :
        Grid(Collective *col);
        ~Grid();

        int   get_ncell();
        float get_Lx();
        float get_Ly();
        float get_Lz();
        float get_dx();
        float get_dy();
        float get_dz();

        int ncx;
        int ncy;
        int ncz;

    private:
        float Lx;
        float Ly;
        float Lz;
        float dx;
        float dy;
        float dz;

        int ncell;

};

#endif
