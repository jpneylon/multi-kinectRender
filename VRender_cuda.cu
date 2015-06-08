#include <algorithm>
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "VRender_cuda_kernel.cuh"
#include "Cloud.h"


int iDivUp( int a, int b ){ return (a % b != 0) ? (a / b + 1) : (a / b); }


// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}


extern "C"
void allocateMemory( Cloud *cloud, int device, cudaExtent volumeSize, uint imageW, uint imageH )
{
    cudaSetDevice(device);

    float cudatime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    checkCudaErrors( cudaMalloc( (void**) &cellStart, cloud->world.count * sizeof(uint) ) );
    checkCudaErrors( cudaMalloc( (void**) &cellEnd, cloud->world.count * sizeof(uint) ) );
    checkCudaErrors( cudaMemset( cellStart, 0xffffffff, cloud->world.count * sizeof(uint) ) );
    checkCudaErrors( cudaMemset( cellEnd, 0xffffffff, cloud->world.count * sizeof(uint) ) );

    checkCudaErrors( cudaMalloc( (void**) &d_red, cloud->world.count ) );
    checkCudaErrors( cudaMalloc( (void**) &d_green, cloud->world.count ) );
    checkCudaErrors( cudaMalloc( (void**) &d_blue, cloud->world.count ) );
    checkCudaErrors( cudaMemset( d_red, 0, cloud->world.count ) );
    checkCudaErrors( cudaMemset( d_green, 0, cloud->world.count ) );
    checkCudaErrors( cudaMemset( d_blue, 0, cloud->world.count ) );

    checkCudaErrors( cudaMemcpyToSymbol( d_pcl, &cloud->pcl, sizeof(PCListData) ) );
    checkCudaErrors( cudaMemcpyToSymbol( d_world, &cloud->world, sizeof(WORLD) ) );

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();

    // RED
    checkCudaErrors( cudaMalloc3DArray( &d_redArray, &channelDesc, volumeSize ) );
    redParams.srcPtr   =   make_cudaPitchedPtr( d_red, volumeSize.width, volumeSize.width, volumeSize.height );
    redParams.dstArray =   d_redArray;
    redParams.extent   =   volumeSize;
    redParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &redParams ) );

    texRed.normalized = true;
    texRed.filterMode = cudaFilterModeLinear;
    texRed.addressMode[0] = cudaAddressModeClamp;
    texRed.addressMode[1] = cudaAddressModeClamp;
    texRed.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texRed, d_redArray, channelDesc ) );

    // GREEN
    checkCudaErrors( cudaMalloc3DArray( &d_greenArray, &channelDesc, volumeSize ) );
    greenParams.srcPtr   =   make_cudaPitchedPtr( d_green, volumeSize.width, volumeSize.width, volumeSize.height );
    greenParams.dstArray =   d_greenArray;
    greenParams.extent   =   volumeSize;
    greenParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &greenParams ) );

    texGreen.normalized = true;
    texGreen.filterMode = cudaFilterModeLinear;
    texGreen.addressMode[0] = cudaAddressModeClamp;
    texGreen.addressMode[1] = cudaAddressModeClamp;
    texGreen.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texGreen, d_greenArray, channelDesc ) );

    // BLUE
    checkCudaErrors( cudaMalloc3DArray( &d_blueArray, &channelDesc, volumeSize ) );
    blueParams.srcPtr   =   make_cudaPitchedPtr( d_blue, volumeSize.width, volumeSize.width, volumeSize.height );
    blueParams.dstArray =   d_blueArray;
    blueParams.extent   =   volumeSize;
    blueParams.kind     =   cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D( &blueParams ) );

    texBlue.normalized = true;
    texBlue.filterMode = cudaFilterModeLinear;
    texBlue.addressMode[0] = cudaAddressModeClamp;
    texBlue.addressMode[1] = cudaAddressModeClamp;
    texBlue.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray( texBlue, d_blueArray, channelDesc ) );

    // OUTPUT BUFFER
    checkCudaErrors( cudaMalloc( (void**) &d_volume, imageW * imageH * 3 ) );
    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 ) );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &cudatime, start, stop );
    printf("\n ||| TIME - GPU Memory Allocation: %f ms\n", cudatime);
}



extern "C"
void updateVRenderColorMaps( Cloud * cloud )
{
    checkCudaErrors( cudaMemcpyToSymbol( d_pcl, &cloud->pcl, sizeof(PCListData) ) );

    h_pos = new float3[cloud->pcl.count];
    h_color = new uint3[cloud->pcl.count];
    checkCudaErrors( cudaMalloc( (void**) &d_pos, cloud->pcl.count * sizeof(float3) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_color, cloud->pcl.count * sizeof(uint3) ) );
    std::copy( cloud->position.begin(), cloud->position.end(), h_pos );
    checkCudaErrors( cudaMemcpy( d_pos, h_pos, cloud->pcl.count * sizeof(float3), cudaMemcpyHostToDevice ) );
    std::copy( cloud->rgb.begin(), cloud->rgb.end(), h_color );
    checkCudaErrors( cudaMemcpy( d_color, h_color, cloud->pcl.count * sizeof(uint3), cudaMemcpyHostToDevice ) );

    uint numThreads, numBlocks;
    computeGridSize( cloud->pcl.count, 256, numBlocks, numThreads);

    checkCudaErrors( cudaMalloc( (void**) &gridHash, cloud->pcl.count * sizeof(uint) ) );
    checkCudaErrors( cudaMalloc( (void**) &gridIndex, cloud->pcl.count * sizeof(uint) ) );
    checkCudaErrors( cudaMemset( gridHash, 0, cloud->pcl.count * sizeof(uint) ) );
    checkCudaErrors( cudaMemset( gridIndex, 0, cloud->pcl.count * sizeof(uint) ) );

    calcHashD<<< numBlocks, numThreads >>>( gridHash,
                                            gridIndex,
                                            d_pos );

    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");

    thrust::sort_by_key(thrust::device_ptr<uint>(gridHash),
                            thrust::device_ptr<uint>(gridHash + cloud->pcl.count),
                            thrust::device_ptr<uint>(gridIndex));

    checkCudaErrors( cudaMemset( cellStart, 0xffffffff, cloud->world.count * sizeof(uint) ) );
    checkCudaErrors( cudaMemset( cellEnd, 0xffffffff, cloud->world.count * sizeof(uint) ) );

    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>( cellStart,
                                                                        cellEnd,
                                                                        gridHash,
                                                                        gridIndex );

    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");


    checkCudaErrors( cudaMemset( d_red, 0, cloud->world.count ) );
    checkCudaErrors( cudaMemset( d_green, 0, cloud->world.count ) );
    checkCudaErrors( cudaMemset( d_blue, 0, cloud->world.count ) );
    cuda_create_color_maps<<< numBlocks, numThreads >>> ( d_pos,
                                                          d_color,
                                                          gridIndex,
                                                          cellStart,
                                                          cellEnd,
                                                          d_red,
                                                          d_green,
                                                          d_blue );
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");

    checkCudaErrors( cudaMemcpy3D( &redParams ) );
    checkCudaErrors( cudaMemcpy3D( &greenParams ) );
    checkCudaErrors( cudaMemcpy3D( &blueParams ) );

    checkCudaErrors( cudaFree( d_pos ) );
    checkCudaErrors( cudaFree( d_color ) );
    checkCudaErrors( cudaFree( gridHash ) );
    checkCudaErrors( cudaFree( gridIndex ) );

    delete [] h_pos;
    delete [] h_color;
}


extern "C"
void freeCudaBuffers()
{
    checkCudaErrors( cudaFree( cellEnd ) );
    checkCudaErrors( cudaFree( cellStart ) );

    checkCudaErrors( cudaUnbindTexture(texRed) );
    checkCudaErrors( cudaFreeArray(d_redArray) );

    checkCudaErrors( cudaUnbindTexture(texGreen) );
    checkCudaErrors( cudaFreeArray(d_greenArray) );

    checkCudaErrors( cudaUnbindTexture(texBlue) );
    checkCudaErrors( cudaFreeArray(d_blueArray) );

    checkCudaErrors( cudaFree( d_red ) );
    checkCudaErrors( cudaFree( d_green ) );
    checkCudaErrors( cudaFree( d_blue ) );

    checkCudaErrors( cudaFree( d_volume ) );
}


extern "C"
void render_kernel( dim3 gridSize, dim3 blockSize,
                    unsigned char *buffer,
                    uint imageW, uint imageH,
                    float dens, float bright, float offset, float scale,
                    float *fps )
{
    float cudatime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    checkCudaErrors( cudaMemset( d_volume, 0, imageW * imageH * 3 ) );
    d_render<<<gridSize,blockSize>>>( d_volume,
                                      imageW, imageH,
                                      dens, bright, offset, scale );
    cudaThreadSynchronize();
    checkCudaErrors( cudaMemcpy( buffer, d_volume, imageW * imageH * 3, cudaMemcpyDeviceToHost ) );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &cudatime, start, stop );
    *fps = cudatime;
}


extern "C"
void copyInvViewMatrix( float *invViewMatrix, size_t sizeofMatrix )
{
    checkCudaErrors( cudaMemcpyToSymbol( c_invViewMatrix, invViewMatrix, sizeofMatrix, 0, cudaMemcpyHostToDevice ) );
}




