#include "Cloud.h"


int iDivUp( int a, int b );


extern "C"
void allocateMemory( Cloud *cloud, int device, cudaExtent volumeSize, uint imageW, uint imageH );


extern "C"
void updateVRenderColorMaps( Cloud * cloud );


extern "C"
    void freeCudaBuffers();


extern "C"
    void render_kernel( dim3 gridSize, dim3 blockSize,
                        unsigned char *buffer,
                        uint imageW, uint imageH,
                        float dens, float bright, float offset, float scale,
                        float *fps );


extern "C"
    void copyInvViewMatrix( float *invViewMatrix,
                            size_t sizeofMatrix);

