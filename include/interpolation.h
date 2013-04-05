#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

#include <stdlib.h>
#include <stdio.h>

#include <iostream>

using std::cout;
using std::endl;

#include "include/fields.h"
#include "include/particles.h"
#include "include/collective.h"
#include "include/grid.h"
#include "include/plasma.h"

#include "include/numerics_gpu.h"
#include "include/numerics_mic.h"

class Interpolation{
    public :
        Interpolation(Collective *col);
        ~Interpolation();

        // Functions
        void Part2Fields(Grid *grid, Fields *fields, Particles *part, Plasma *plasma);
        void Fields2Part(Grid *grid, Fields *fields, Particles *part);

        void Part2Fields_cpu2cpu(Grid *grid, Fields *fields, Particles *part, Plasma *plasma);

        // GPU
        void Part2Fields_gpu(Grid *grid, Fields *fields, Particles *part, Plasma *plasma);
        void Fields2Part_gpu(Grid *grid, Fields *fields, Particles *part);
        // MIC
        void Part2Fields_mic(Grid *grid, Fields *fields, Particles *part, Plasma *plasma);
        void Fields2Part_mic(Grid *grid, Fields *fields, Particles *part);

    private :
        int interptype;
        int acc;
};

#endif
