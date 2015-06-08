#ifndef __VRENDER_CLASS_H__
#define __VRENDER_CLASS_H__

// CUDA utilities and system includes
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

// CUDA Includes
#include <vector_functions.h>
#include <driver_functions.h>

#include "Cloud.h"

#define PI 3.14159f
#define BUFFER_SIZE 720
#define FPS_SIZE 100
#define RENDER_RESOLUTION 5


class VRender
{
    public:
        VRender();
        ~VRender();

        //functions
        unsigned char *get_vrender_buffer( Cloud *cloud );
        char *get_vrender_fps(){ return fps_text; };

        void set_vrender_parameters( float r_dens, float r_bright, float r_offset, float r_scale );
        void set_vrender_rotation( float dx, float dy );
        void set_vrender_translation( float dx, float dy );
        void set_vrender_zoom( float dy );

        void update_color_maps( Cloud *cloud );
        void allocate_memory( Cloud *cloud, int device );

        int get_width()
        {
            return width;
        };
        int get_height()
        {
            return height;
        };
        float get_density()
        {
            return density;
        };
        float get_brightness()
        {
            return brightness;
        };
        float get_offset()
        {
            return transferOffset;
        };
        float get_scale()
        {
            return transferScale;
        };
        float get_last_x()
        {
            return last_x;
        };
        float get_last_y()
        {
            return last_y;
        };

        void set_width( int i )
        {
            width = i;
        };
        void set_height( int i )
        {
            height = i;
        };
        void set_density( float v )
        {
            density = v;
        };
        void set_brightness( float v )
        {
            brightness = v;
        };
        void set_offset( float v )
        {
            transferOffset = v;
        };
        void set_scale( float v )
        {
            transferScale = v;
        };
        void set_last_x( float v )
        {
            last_x = v;
        };
        void set_last_y( float v )
        {
            last_y = v;
        };

    protected:
       //functions
        void setInvViewMatrix();
        void render( Cloud *cloud );

        void translateMat( float *matrix, float3 translation );
        void rotMat( float *matrix, float3 axis, float theta, float3 center );
        void multiplyModelViewMatrix( float *trans );
        void transformModelViewMatrix();

        //variables
        uint width, height;
        unsigned char cycle;

        uint frame_counter;
        uint swap;

        dim3 blockSize;
        dim3 gridSize;

        cudaExtent volumeSize;

        float3 viewRotation;
        float3 viewTranslation;

        float invViewMatrix[12];
        float identityMatrix[16];
        float modelViewMatrix[16];

        float density;
        float brightness;
        float transferOffset;
        float transferScale;
        float last_x, last_y;

        uint  fps_idx;
        float *fps_frames;
        char  *fps_text;

        uint world_size;

        unsigned char *render_buf;
};


#endif // __VRENDER_CLASS_H__
