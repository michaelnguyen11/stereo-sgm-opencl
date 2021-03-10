#include "census_transform.h"

namespace sgmcl
{

    CensusTransform::CensusTransform(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx), m_cl_device(device)
    {
    }

    CensusTransform::~CensusTransform()
    {
        if (m_census_kernel)
        {
            clReleaseKernel(m_census_kernel);
            m_census_kernel = nullptr;
        }
    }

    void CensusTransform::enqueue(const DeviceBuffer<uint8_t> &src,
                                  DeviceBuffer<uint32_t> &feature_buffer,
                                  int width,
                                  int height,
                                  int pitch,
                                  cl_command_queue stream)
    {
        if (m_census_kernel == nullptr)
        {
            //resource reading
            std::ifstream fileInput;
            fileInput.open("data/ocl/census.cl");
            std::string kernel{std::istreambuf_iterator<char>(fileInput),
                               std::istreambuf_iterator<char>()};
            m_program.init(m_cl_ctx, m_cl_device, kernel);

            m_census_kernel = m_program.getKernel("census_transform_kernel");
        }

        cl_int err;
        err = clSetKernelArg(m_census_kernel, 0, sizeof(cl_mem), &feature_buffer.data());
        err = clSetKernelArg(m_census_kernel, 1, sizeof(cl_mem), &src.data());
        err = clSetKernelArg(m_census_kernel, 2, sizeof(width), &width);
        err = clSetKernelArg(m_census_kernel, 3, sizeof(height), &height);
        err = clSetKernelArg(m_census_kernel, 4, sizeof(pitch), &pitch);

        const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
        const int height_per_block = LINES_PER_BLOCK;

        //setup kernels
        size_t global_size[2] = {
            (size_t)((width + width_per_block - 1) / width_per_block * BLOCK_SIZE),
            (size_t)((height + height_per_block - 1) / height_per_block)};
        size_t local_size[2] = {BLOCK_SIZE, 1};
        err = clEnqueueNDRangeKernel(stream,
                                     m_census_kernel,
                                     2,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error enequeuing census kernel");
    }
} // namespace sgmcl
