#include "include/plasma.h"

Plasma::Plasma(Collective *col){
    qom = col->qom;
    u0  = col->u0;
    v0  = col->v0;
    w0  = col->w0;
    ut  = col->ut;
    vt  = col->vt;
    wt  = col->wt;
    wpi = col->wpi;
}

Plasma::~Plasma(){
}
