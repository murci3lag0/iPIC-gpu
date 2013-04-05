/*  -----------------------------------------------------------------
 *                                                                    *
 *  Vlasov-Ampere GPU code for PIC in 3D                              *
 *                                                                    *
 *  Copyrigth (c) CmPA, Mathematics Department, K.U. Leuven           *
 *                                                                    *
 *  Based on G. Lapenta's matlab example code                         *
 *  GPU adaptation (v.1, 1D) by: J.Deca, 21 June 2012                 *
 *  Data structure and 3D extension (v.2) by: J. Amaya , 26 Feb. 2013 *
 *                                                                    *
 *  ----------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

#include "include/fields.h"
#include "include/particles.h"
#include "include/collective.h"
#include "include/grid.h"
#include "include/numerics.h"
#include "include/interpolation.h"

int main(int argc, char* argv[]) {

    // Get input
    string inputfile = "input.inp";

    // Construct the classes
    Collective *col    = new Collective(inputfile);
    Grid       *grid   = new Grid(col);
    Particles  *part   = new Particles(col, grid);
    Fields     *fields = new Fields(col);
    Plasma     *plasma = new Plasma(col);

    // Fields ans particles initialization
    fields -> InitFieldsZero(grid);
    part   -> GenerateParticles(col,grid);

    // Select the numerics and the interpolation methods
    Numerics      numerics(col);
    Interpolation interp(col);

    // Add perturbation to the particles
    if (col->ipert > 0 && col->get_acc()==1 ){
        part-> AddPerturbation(col,grid);
        interp.Part2Fields_cpu2cpu(grid, fields, part, plasma);
    }

    // Init ACCELERATOR vectors
    // /!\ Maybe this needs to be moved up, just after input file reading
    numerics.InitHardware(grid, fields, part);

#ifdef __GPU__
    if (col->get_acc()==1) MemCpy_cpu2gpu(grid, fields, part);
#endif

    for (int i = 0; i < col->niter; i++) {

        if ((i+1) % 10 == 0) cout << "it : " << (i+1) << " / " << col->niter << endl;

        numerics.UpdateFields   (grid, fields);
        numerics.MoveParticles  (grid, part);
        interp.  Part2Fields    (grid, fields, part, plasma);
        numerics.CalculateFields(grid, fields);
        interp.  Fields2Part    (grid, fields, part);
        numerics.UpdateVelocity (grid, part, plasma);

    }

#ifdef __GPU__
    if (col->get_acc()==1) MemCpy_gpu2cpu(grid, fields, part);
#endif

    numerics.StopHardware(fields, part);

    return(0);

}
