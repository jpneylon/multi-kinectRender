#ifndef __CLOUD_CLASS_H__
#define __CLOUD_CLASS_H__

#include <vector>



typedef struct{
    uint count;
    float3 max;
    float3 min;
    float3 dimension;
} PCListData;


typedef struct {
    uint count;
    uint3 size;
    float3 origin;
    float3 min;
    float3 max;
    float3 resolution;
    float3 dimension;
} WORLD;


class Cloud
{
  public:
    Cloud()
    {
        pcl.count = 0;
        pcl.max.x = -999999;
        pcl.max.y = -999999;
        pcl.max.z = -999999;
        pcl.min.x = 999999;
        pcl.min.y = 999999;
        pcl.min.z = 999999;
    }
    std::vector<float3> position;
    std::vector<uint3> rgb;
    PCListData pcl;
    WORLD world;
};


#endif // __CLOUD_CLASS_H__
