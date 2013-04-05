#ifndef __NUMERICS_H__
#define __NUMERICS_H__

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

class Numerics{
    public :
        Numerics(Collective *col);
        ~Numerics();

        // Functions
        void MoveParticles  (Grid *grid, Particles *part);
        void CalculateFields(Grid *grid, Fields *fields);
        void UpdateFields   (Grid *grid, Fields *fields);
        void UpdateVelocity (Grid *grid, Particles *part, Plasma *plasma);

        void InitHardware(Grid *grid, Fields *fields, Particles *part);
        void StopHardware(Fields *fields, Particles *part);

        // GPU
        void UpdateFields_gpu   (Grid *grid, Fields *fields);
        void MoveParticles_gpu  (Grid *grid, Particles *part);
        void CalculateFields_gpu(Grid *grid, Fields *fields);
        void UpdateVelocity_gpu (Grid *grid, Particles *part, Plasma *plasma);
        // MIC
        void UpdateFields_mic   (Grid *grid, Fields *fields);
        void MoveParticles_mic  (Grid *grid, Particles *part);
        void CalculateFields_mic(Grid *grid, Fields *fields);
        void UpdateVelocity_mic (Grid *grid, Particles *part, Plasma *plasma);

        float dt;

    private :
        int acc;
        int iorder;

};

#endif
