
#ifndef __PARTICLES_H__
#define __PARTICLES_H__

#include <stdlib.h>
#include <math.h>

#include "include/constants.h"

#include "include/collective.h"
#include "include/grid.h"

class Particles {
    public :
        Particles(Collective *col, Grid *grid);
        ~Particles();

        // Functions:
        void AllocateParticles();
        void GenerateParticles(Collective *col, Grid *grid);
        void AddPerturbation(Collective *col, Grid *grid);

        void PartFieldsZero();
        void distx_random(Grid *grid);
        void distu_randomthermal(Collective *col, Grid *grid);
        void distu_maxwellian(Collective *col, Grid *grid);

        // Public vaiables:
        int npart;

        // CPU pointers
        float *u;
        float *v;
        float *w;
        float *x;
        float *y;
        float *z;
        float *Ex;
        float *Ey;
        float *Ez;

        // GPU pointers
        float *dev_u;
        float *dev_v;
        float *dev_w;
        float *dev_x;
        float *dev_y;
        float *dev_z;
        float *dev_Ex;
        float *dev_Ey;
        float *dev_Ez;

    private:
        int ppc;

};

#endif
