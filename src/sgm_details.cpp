#include "sgm_details.h"

namespace sgmcl
{
    SGMDetails::SGMDetails(cl_context ctx, cl_device_id device)
        : m_cl_context(ctx), m_cl_device_id(device)
    {
    }

    SGMDetails::~SGMDetails()
    {
        if (m_kernel_check_consistency)
        {
            clReleaseKernel(m_kernel_check_consistency);
            m_kernel_check_consistency = nullptr;
        }
        if (m_kernel_median)
        {
            clReleaseKernel(m_kernel_median);
            m_kernel_median = nullptr;
        }
        if (m_kernel_disp_corr)
        {
            clReleaseKernel(m_kernel_disp_corr);
            // clReleaseKernel(m_kernel_cast_16uto8u);
            m_kernel_disp_corr = nullptr;
            // m_kernel_cast_16uto8u = nullptr;
        }
    }

    void SGMDetails::median_filter(const DeviceBuffer<uint16_t> &d_src,
                                   const DeviceBuffer<uint16_t> &d_dst,
                                   int width,
                                   int height,
                                   int pitch,
                                   cl_command_queue stream)
    {
        if (nullptr == m_kernel_median)
        {
            std::ifstream fileInput;
            fileInput.open("data/ocl/median_filter.cl");
            std::string kernel_src{std::istreambuf_iterator<char>(fileInput),
                                   std::istreambuf_iterator<char>()};
            m_program_median.init(m_cl_context, m_cl_device_id, kernel_src);
            m_kernel_median = m_program_median.getKernel("median3x3");
        }
        cl_int err = clSetKernelArg(m_kernel_median,
                                    0,
                                    sizeof(cl_mem),
                                    &d_src.data());
        err = clSetKernelArg(m_kernel_median, 1, sizeof(cl_mem), &d_dst.data());
        err = clSetKernelArg(m_kernel_median, 2, sizeof(width), &width);
        err = clSetKernelArg(m_kernel_median, 3, sizeof(height), &height);
        err = clSetKernelArg(m_kernel_median, 4, sizeof(pitch), &pitch);

        static constexpr int SIZE = 16;
        size_t local_size[2] = {SIZE, SIZE};
        size_t global_size[2] = {
            ((width + SIZE - 1) / SIZE) * local_size[0],
            ((height + SIZE - 1) / SIZE) * local_size[1]};

        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel_median,
                                     2,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");
    }

    void SGMDetails::check_consistency(DeviceBuffer<uint16_t> &d_left_disp,
                                       const DeviceBuffer<uint16_t> &d_right_disp,
                                       const DeviceBuffer<uint8_t> &d_src_left,
                                       int width,
                                       int height,
                                       int src_pitch,
                                       int dst_pitch,
                                       bool subpixel,
                                       int LR_max_diff,
                                       cl_command_queue stream)
    {
        if (nullptr == m_kernel_check_consistency)
        {
            std::ifstream fileInput1, fileInput2;
            fileInput1.open("data/ocl/inttypes.cl");
            fileInput2.open("data/ocl/check_consistency.cl");
            std::string src1{std::istreambuf_iterator<char>(fileInput1),
                             std::istreambuf_iterator<char>()};
            std::string src2{std::istreambuf_iterator<char>(fileInput2),
                             std::istreambuf_iterator<char>()};
            std::string kernel_src = src1 + src2;

            std::string kernel_SUBPIXEL_SHIFT = "#define SUBPIXEL_SHIFT " + std::to_string(SubpixelShift()) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@SUBPIXEL_SHIFT@"), kernel_SUBPIXEL_SHIFT);

            m_program_check_consistency.init(m_cl_context, m_cl_device_id, kernel_src);
            m_kernel_check_consistency = m_program_check_consistency.getKernel("check_consistency_kernel");
        }

        static constexpr int SIZE = 16;
        size_t local_size[2] = {SIZE, SIZE};
        size_t global_size[2] = {
            ((width + SIZE - 1) / SIZE) * local_size[0],
            ((height + SIZE - 1) / SIZE) * local_size[1]};
        cl_int err = clSetKernelArg(m_kernel_check_consistency,
                                    0,
                                    sizeof(cl_mem),
                                    &d_left_disp.data());
        err = clSetKernelArg(m_kernel_check_consistency, 1, sizeof(cl_mem), &d_right_disp.data());
        err = clSetKernelArg(m_kernel_check_consistency, 2, sizeof(cl_mem), &d_src_left.data());
        err = clSetKernelArg(m_kernel_check_consistency, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel_check_consistency, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel_check_consistency, 5, sizeof(src_pitch), &src_pitch);
        err = clSetKernelArg(m_kernel_check_consistency, 6, sizeof(dst_pitch), &dst_pitch);
        int sub_pixel_int = subpixel ? 1 : 0;
        err = clSetKernelArg(m_kernel_check_consistency, 7, sizeof(sub_pixel_int), &sub_pixel_int);
        err = clSetKernelArg(m_kernel_check_consistency, 8, sizeof(LR_max_diff), &LR_max_diff);

        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel_check_consistency,
                                     2,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");
    }

    void SGMDetails::correct_disparity_range(DeviceBuffer<uint16_t> &d_disp,
                                             int width,
                                             int height,
                                             int pitch,
                                             bool subpixel,
                                             int min_disp,
                                             cl_command_queue stream)
    {
        if (!subpixel && min_disp == 0)
        {
            return;
        }

        if (nullptr == m_kernel_disp_corr)
        {
            initDispRangeCorrection();
        };

        const int scale = subpixel ? SubpixelScale() : 1;
        const int min_disp_scaled = min_disp * scale;
        const int invalid_disp_scaled = (min_disp - 1) * scale;

        cl_int err = clSetKernelArg(m_kernel_disp_corr,
                                    0,
                                    sizeof(cl_mem),
                                    &d_disp.data());
        err = clSetKernelArg(m_kernel_disp_corr, 1, sizeof(width), &width);
        err = clSetKernelArg(m_kernel_disp_corr, 2, sizeof(height), &height);
        err = clSetKernelArg(m_kernel_disp_corr, 3, sizeof(pitch), &pitch);
        err = clSetKernelArg(m_kernel_disp_corr, 4, sizeof(min_disp_scaled), &min_disp_scaled);
        err = clSetKernelArg(m_kernel_disp_corr, 5, sizeof(invalid_disp_scaled), &invalid_disp_scaled);

        static constexpr int SIZE = 16;
        size_t local_size[2] = {SIZE, SIZE};
        size_t global_size[2] = {
            ((width + SIZE - 1) / SIZE) * local_size[0],
            ((height + SIZE - 1) / SIZE) * local_size[1]};

        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel_disp_corr,
                                     2,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error enequeuing correct disparity range kernel");
    }

    void SGMDetails::initDispRangeCorrection()
    {
        std::ifstream fileInput;
        fileInput.open("data/ocl/inttypes.cl");
        std::string kernel_src{std::istreambuf_iterator<char>(fileInput),
                               std::istreambuf_iterator<char>()};

        fileInput.close();
        fileInput.open("data/ocl/correct_disparity_range.cl");
        std::string kernel_src2{std::istreambuf_iterator<char>(fileInput),
                                std::istreambuf_iterator<char>()};

        kernel_src = kernel_src + kernel_src2;
        m_program_disp_corr.init(m_cl_context, m_cl_device_id, kernel_src);

        m_kernel_disp_corr = m_program_disp_corr.getKernel("correct_disparity_range_kernel");
        // m_kernel_cast_16uto8u = m_program_disp_corr.getKernel("cast_16bit_8bit_array_kernel");
    }
} // namespace sgmcl
