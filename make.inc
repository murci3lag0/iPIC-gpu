CPP=nvcc
NVCC=nvcc

NVARCH=sm_21

# DACC: Definition of the accelerator
# Adds certain subroutines depending on the selected
# accelerator type: __NONE__, __GPU__, __MIC__, __MPI__, etc.
ACC=__GPU__

# ATOMICGPU: Select the origin of the atomic operations
# If ATOMICGPU = -DOLDGPU atomic operations are handled using 
# an in-house developed atomic operations kernel. If no
# value is given, the code uses the intrinsic operations.
ATOMICGPU=

# CURDIR: Location of the source file directory
# If no value is given, makefile sets the current PWD directory
# to this variable.
#CURDIR=.

LDFLAGS=
NVCCFLAGS=-arch=$(NVARCH) $(ATOMICGPU)
DACC=-D$(ACC)
