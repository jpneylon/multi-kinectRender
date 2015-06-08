#include "VRender.h"
#include "VRender_cuda.cuh"

VRender::VRender()
{
    volumeSize = make_cudaExtent( 256, 256, 256 );

    width = BUFFER_SIZE;
    height = BUFFER_SIZE;

    blockSize.x = 16;
    blockSize.y = 16;

    viewRotation = make_float3( 90, 180, 0 );
    viewTranslation = make_float3( 0, 0, 3 );

    density = 0.25f;
    brightness = 2.5f;
    transferOffset = 0.f;
    transferScale = 1.f;

    fps_idx = 0;
    fps_frames = new float[FPS_SIZE];
    memset( fps_frames, 0, FPS_SIZE * sizeof(float) );
    fps_text = new char[16];
    sprintf(fps_text,"0.0");

    cycle = 0;
    frame_counter = 0;
    swap = 0;
}
VRender::~VRender()
{
    freeCudaBuffers();
    delete [] render_buf;
    delete [] fps_text;
    delete [] fps_frames;
}


void
VRender::set_vrender_parameters( float r_dens, float r_bright, float r_offset, float r_scale )
{
    density = r_dens;
    brightness = r_bright;
    transferOffset = r_offset;
    transferScale = r_scale;
}
void
VRender::set_vrender_rotation( float dx, float dy )
{
    viewRotation.x -= dy;
    viewRotation.y += dx;

    transformModelViewMatrix();
    setInvViewMatrix();
}
void
VRender::set_vrender_translation( float dx, float dy )
{
    viewTranslation.x += dx;
    viewTranslation.y += dy;

    transformModelViewMatrix();
    setInvViewMatrix();
}
void
VRender::set_vrender_zoom( float dy )
{
    viewTranslation.z += dy;

    transformModelViewMatrix();
    setInvViewMatrix();
}

void
VRender::setInvViewMatrix()
{
    invViewMatrix[0]     =   modelViewMatrix[0];    //rot_x.x
    invViewMatrix[1]     =   modelViewMatrix[1];    //rot_x.y
    invViewMatrix[2]     =   modelViewMatrix[2];    //rot_x.z
    invViewMatrix[3]     =   modelViewMatrix[3];    //trans.x
    invViewMatrix[4]     =   modelViewMatrix[4];    //rot_y.x
    invViewMatrix[5]     =   modelViewMatrix[5];    //rot_y.y
    invViewMatrix[6]     =   modelViewMatrix[6];    //rot_y.z
    invViewMatrix[7]     =   modelViewMatrix[7];    //trans.y
    invViewMatrix[8]     =   modelViewMatrix[8];    //rot_z.x
    invViewMatrix[9]     =   modelViewMatrix[9];    //rot_z.y
    invViewMatrix[10]    =   modelViewMatrix[10];   //rot_z.z
    invViewMatrix[11]    =   modelViewMatrix[11];   //trans.z

    copyInvViewMatrix( invViewMatrix, sizeof(float4)*3 );
}


void
VRender::translateMat( float *matrix, float3 translation )
{
    float3 start = make_float3( matrix[3], matrix[7], matrix[11] );

    start.x -= translation.x;
    start.y -= translation.y;
    start.z -= translation.z;

    float3 new_origin;
    new_origin.x = matrix[0] * start.x + matrix[1] * start.y + matrix[2] * start.z;
    new_origin.y = matrix[4] * start.x + matrix[5] * start.y + matrix[6] * start.z;
    new_origin.z = matrix[8] * start.x + matrix[9] * start.y + matrix[10] * start.z;

    matrix[3] = new_origin.x;
    matrix[7] = new_origin.y;
    matrix[11] = new_origin.z;
}

void
VRender::rotMat( float *matrix, float3 axis, float theta, float3 center )
{
    for (int v=0; v<3; v++)
    {
        float3 rot = make_float3( (theta * PI / 180) * axis.x,
                                  (theta * PI / 180) * axis.y,
                                  (theta * PI / 180) * axis.z );

        float3 start = make_float3( matrix[0 + v*4] - center.x,
                                    matrix[1 + v*4] - center.y,
                                    matrix[2 + v*4] - center.z );

        float3 inter_x = make_float3 ( start.x,
                                       start.y * cos(rot.x) - start.z * sin(rot.x),
                                       start.z * cos(rot.x) + start.y * sin(rot.x) );

        float3 inter_y = make_float3 ( inter_x.x * cos(rot.y) + inter_x.z * sin(rot.y),
                                       inter_x.y,
                                       inter_x.z * cos(rot.y) - inter_x.x * sin(rot.y) );

        float3 inter_z = make_float3 ( inter_y.x * cos(rot.z) - inter_y.y * sin(rot.z),
                                       inter_y.y * cos(rot.z) + inter_y.x * sin(rot.z),
                                       inter_y.z );

        matrix[0 + v*4] = inter_z.x + center.x;
        matrix[1 + v*4] = inter_z.y + center.y;
        matrix[2 + v*4] = inter_z.z + center.z;
    }
}

void
VRender::multiplyModelViewMatrix( float *trans )
{
    float *result;
    result = new float[16];

    for (int r=0; r<3; r++)
        for (int c=0; c<3; c++)
        {
            float4 row = make_float4( trans[0 + 4*r],
                                      trans[1 + 4*r],
                                      trans[2 + 4*r],
                                      trans[3 + 4*r] );

            float4 col = make_float4( modelViewMatrix[0 + c],
                                      modelViewMatrix[4 + c],
                                      modelViewMatrix[8 + c],
                                      modelViewMatrix[12 + c] );

            result[c + 4*r] = row.x * col.x +
                              row.y * col.y +
                              row.z * col.z +
                              row.w * col.w;
        }

    memcpy( modelViewMatrix, result, 16 * sizeof(float) );
    delete [] result;
}

void
VRender::transformModelViewMatrix()
{
    float *matrix;
    matrix = new float[16];
    memcpy( matrix, identityMatrix, 16 * sizeof(float) );

    rotMat( matrix, make_float3(1,0,0), -viewRotation.x, make_float3(0,0,0) );
    rotMat( matrix, make_float3(0,1,0), -viewRotation.y, make_float3(0,0,0) );
    translateMat( matrix, viewTranslation );

    memcpy( modelViewMatrix, matrix, 16 * sizeof(float) );

    delete [] matrix;
}

void
VRender::update_color_maps( Cloud *cloud )
{
    updateVRenderColorMaps( cloud );
}

void
VRender::render( Cloud *cloud )
{
    render_kernel( gridSize, blockSize, render_buf, width, height, density, brightness, transferOffset, transferScale, &fps_frames[fps_idx] );

    float time = fps_frames[0];
    for (uint f = 1; f < FPS_SIZE; f++)
        time += fps_frames[f];
    sprintf(fps_text,"%f", time / FPS_SIZE );
    fps_idx = (fps_idx + 1) % FPS_SIZE;
}

unsigned char *
VRender::get_vrender_buffer( Cloud *cloud )
{
    render( cloud );
    return (render_buf);
}


void
VRender::allocate_memory( Cloud *cloud, int device )
{
    volumeSize.width = cloud->world.size.x;
    volumeSize.height = cloud->world.size.y;
    volumeSize.depth = cloud->world.size.z;

    render_buf = new unsigned char[ height * width * 3 ];

    allocateMemory( cloud, device, volumeSize, width, height );

    gridSize = dim3( iDivUp(width, blockSize.x), iDivUp(height, blockSize.y) );

    memset( identityMatrix, 0, 16 * sizeof(float ) );
    identityMatrix[0]  = 1;
    identityMatrix[5]  = 1;
    identityMatrix[10] = 1;
    identityMatrix[15] = 1;

    memcpy( modelViewMatrix, identityMatrix, 16 * sizeof(float ) );
    transformModelViewMatrix();
    setInvViewMatrix();
}




