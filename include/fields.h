
#ifndef __FIELDS_H__
#define __FIELDS_H__

#include "include/collective.h"
#include "include/grid.h"

class Fields {
    public :
        Fields(Collective *col);
        ~Fields();

        void allocatefields(int nx, int ny, int nz);
        void InitFieldsZero(Grid *grid);

        // CPU pointers
        float *Jx;
        float *Jy;
        float *Jz;

        float *Ex;
        float *Ey;
        float *Ez;

        // GPU pointers
        float *dev_Jx;
        float *dev_Jy;
        float *dev_Jz;

        float *dev_Ex;
        float *dev_Ey;
        float *dev_Ez;

    private :
};

#endif
