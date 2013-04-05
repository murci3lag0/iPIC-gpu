#ifndef __GPUTOOLS_CUH__
#define __GPUTOOLS_CUH__

#ifdef __GPU__

#include <stdio.h>
#include <cuda_runtime_api.h>

// GPU error handing
static void HandleError(cudaError_t err, const char *file, int line) {
    if ( err != cudaSuccess ) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) ( HandleError(err,__FILE__,__LINE__) )

// define atomicAdd for floats, since not supported with -arch=sm_11
#ifdef OLDGPU
__device__ inline void atomicFloatAdd(float *address, float val) {
    int tmp0 = *address;
    int i_val = __float_as_int(val + __int_as_float(tmp0));
    int tmp1;

    while((tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0) {
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}
#define ATOMADD(vec ,val) atomicFloatAdd(vec, val)
#else
#define ATOMADD(vec ,val) atomicAdd(vec, val)
#endif

#endif // __GPU__ definition

#endif // Unit definition
