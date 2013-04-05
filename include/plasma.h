
#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "include/collective.h"

class Plasma {
    public :
        // Constructor and destructor
        Plasma(Collective *col);
        ~Plasma();

        // Functions
        // Plasma properties:
        float qom;
        float u0;
        float v0;
        float w0;
        float ut;
        float vt;
        float wt;
        float wpi;

    private:
};

#endif
