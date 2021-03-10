#pragma once
#include "common.h"
#include "device_buffer.h"
#include "device_kernel.h"
#include <regex>

#include <fstream>

namespace sgmcl
{

    class CensusTransform
    {
        static constexpr unsigned int BLOCK_SIZE = 128;
        static constexpr unsigned int WINDOW_WIDTH = 9;
        static constexpr unsigned int WINDOW_HEIGHT = 7;
        static constexpr unsigned int LINES_PER_BLOCK = 16;

    public:
        CensusTransform(cl_context ctx,
                        cl_device_id device);
        ~CensusTransform();

        void enqueue(
            const DeviceBuffer<uint8_t> &src,
            DeviceBuffer<uint32_t> &feature_buffer,
            int width,
            int height,
            int pitch,
            cl_command_queue stream);

    private:
        DeviceProgram m_program;
        cl_context m_cl_ctx = nullptr;
        cl_device_id m_cl_device = nullptr;
        cl_kernel m_census_kernel = nullptr;
    };
} // namespace sgmcl
