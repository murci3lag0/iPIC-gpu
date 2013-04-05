#ifndef __NUMERICS_MIC__
#define __NUMERICS_MIC__

#include "include/grid.h"
#include "include/fields.h"
#include "include/particles.h"

void Init_mic(Grid *grid, Fields *fields, Particles *part);
void Stop_mic();

#endif
