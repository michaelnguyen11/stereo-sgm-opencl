/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
#include "device_buffer.h"
#include "common.h"
#include "device_kernel.h"
#include <regex>
#include <vector>

namespace sgmcl
{

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    class VerticalPathAggregation
    {
    public:
        VerticalPathAggregation(cl_context ctx, cl_device_id device);
        ~VerticalPathAggregation();

        void enqueue(DeviceBuffer<uint8_t> &dest,
                     const DeviceBuffer<uint32_t> &left,
                     const DeviceBuffer<uint32_t> &right,
                     int width,
                     int height,
                     unsigned int p1,
                     unsigned int p2,
                     int min_disp,
                     cl_command_queue stream);

    private:
        void init();
        static constexpr unsigned int WARP_SIZE = 32;
        static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;
        static constexpr unsigned int DP_BLOCK_SIZE = 16u;

        DeviceProgram m_program;
        cl_context m_cl_ctx;
        cl_device_id m_cl_device;
        cl_kernel m_kernel = nullptr;
    };

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    class HorizontalPathAggregation
    {
    public:
        HorizontalPathAggregation(cl_context ctx, cl_device_id device);
        ~HorizontalPathAggregation();

        void enqueue(DeviceBuffer<uint8_t> &dest,
                     const DeviceBuffer<uint32_t> &left,
                     const DeviceBuffer<uint32_t> &right,
                     int width,
                     int height,
                     unsigned int p1,
                     unsigned int p2,
                     int min_disp,
                     cl_command_queue stream);

    private:
        void init();

        DeviceProgram m_program;
        cl_context m_cl_ctx = nullptr;
        cl_device_id m_cl_device = nullptr;
        cl_kernel m_kernel = nullptr;

        static constexpr unsigned int WARP_SIZE = 32;
        static constexpr unsigned int DP_BLOCK_SIZE = 8u;
        static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;
        static constexpr unsigned int WARPS_PER_BLOCK = 4u;
        static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;
    };

    template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
    struct ObliquePathAggregation
    {
    public:
        ObliquePathAggregation(cl_context ctx, cl_device_id device);
        ~ObliquePathAggregation();

        void enqueue(DeviceBuffer<uint8_t> &dest,
                     const DeviceBuffer<uint32_t> &left,
                     const DeviceBuffer<uint32_t> &right,
                     int width,
                     int height,
                     unsigned int p1,
                     unsigned int p2,
                     int min_disp,
                     cl_command_queue stream);

    private:
        DeviceProgram m_program;
        cl_context m_cl_ctx = nullptr;
        cl_device_id m_cl_device = nullptr;
        cl_kernel m_kernel = nullptr;

        static constexpr unsigned int WARP_SIZE = 32;
        static constexpr unsigned int DP_BLOCK_SIZE = 16u;
        static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

        void init();
    };

    template <size_t MAX_DISPARITY>
    class PathAggregation
    {
    public:
        PathAggregation(cl_context ctx, cl_device_id device);
        ~PathAggregation();

        const DeviceBuffer<uint8_t> &get_output() const;

        void enqueue(
            const DeviceBuffer<uint32_t> &left,
            const DeviceBuffer<uint32_t> &right,
            int width,
            int height,
            PathType path_type,
            unsigned int p1,
            unsigned int p2,
            int min_disp,
            cl_command_queue stream);

    private:
        static const unsigned int MAX_NUM_PATHS = 8;

        DeviceBuffer<uint8_t> m_cost_buffer;
        std::vector<DeviceBuffer<uint8_t>> m_sub_buffers;
        cl_command_queue m_streams[MAX_NUM_PATHS];

        //opencl stuff
        VerticalPathAggregation<-1, MAX_DISPARITY> m_down2up;
        VerticalPathAggregation<1, MAX_DISPARITY> m_up2down;
        HorizontalPathAggregation<-1, MAX_DISPARITY> m_right2left;
        HorizontalPathAggregation<1, MAX_DISPARITY> m_left2right;
        ObliquePathAggregation<1, 1, MAX_DISPARITY> m_upleft2downright;
        ObliquePathAggregation<-1, 1, MAX_DISPARITY> m_upright2downleft;
        ObliquePathAggregation<-1, -1, MAX_DISPARITY> m_downright2upleft;
        ObliquePathAggregation<1, -1, MAX_DISPARITY> m_downleft2upright;
    };
} // namespace sgmcl
