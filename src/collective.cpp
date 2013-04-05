#include "include/collective.h"

Collective::Collective(string filename){
    inputfile = filename;
    readinput(inputfile);
}

void Collective::readinput(string inputfile){

    ConfigFile config(inputfile);

    // Geometry:
    Lx    = config.read<float>("Lx");
    Ly    = config.read<float>("Ly");
    Lz    = config.read<float>("Lz");
    ncx   = config.read<int>  ("ncx");
    ncy   = config.read<int>  ("ncy");
    ncz   = config.read<int>  ("ncz");
    npart = config.read<int>  ("npart");

    // Time:
    niter = config.read<int>  ("niter");
    dt    = config.read<float>("dt");

    // Numerics:
    acc        = config.read<int>("accelerator");
    interptype = config.read<int>("interpolator");
    iorder     = config.read<int>("order");

    // Plasma properties:
    udistri = config.read<int>("udist");
    xdistri = config.read<int>("xdist");
    
    qom = config.read<float>("qom");
    u0  = config.read<float>("u0");
    v0  = config.read<float>("v0");
    w0  = config.read<float>("w0");
    ut  = config.read<float>("ut");
    vt  = config.read<float>("vt");
    wt  = config.read<float>("wt");
    wpi = config.read<float>("wpi");

    // Perturbations to the plasma
    ipert  = config.read<int>  ("ipert");
    u_pert = config.read<float>("u_pert");
    x_pert = config.read<float>("x_pert");
}

int Collective::get_acc(){
    return acc;
}

int Collective::get_interptype(){
    return interptype;
}

int Collective::get_iorder(){
    return iorder;
}

int Collective::get_xdistri(){
    return xdistri;
}

int Collective::get_udistri(){
    return udistri;
}

Collective::~Collective(){
}
