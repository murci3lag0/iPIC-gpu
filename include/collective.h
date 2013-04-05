
#ifndef __COLLECTIVE_H__
#define __COLLECTIVE_H__

#include <stdio.h>

#include <string>

using std::string;

#include "include/ConfigFile.h"

class Collective {
    public :
        // Constructor and destructor
        Collective(string inputfile);
        ~Collective();

        // Functions
        void readinput(string inputfile);
        int  get_acc();
        int  get_interptype();
        int  get_iorder();
        int  get_udistri();
        int  get_xdistri();

        // Geometry:
        float Lx;
        float Ly;
        float Lz;
        int   ncx;
        int   ncy;
        int   ncz;
        int   npart;

        // Time:
        int   niter;
        float dt;

        // Plasma properties:
        float qom;
        float u0;
        float v0;
        float w0;
        float ut;
        float vt;
        float wt;
        float wpi;

        // Perturbations to the plasma
        int   ipert;
        float u_pert;
        float x_pert;

    private:
        string inputfile;

        int    acc;
        int    interptype;
        int    iorder;
        int    udistri;
        int    xdistri;
};

#endif
