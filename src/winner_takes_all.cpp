#include "winner_takes_all.h"
#include <regex>
#include <fstream>

//for debugging
#include <opencv2/opencv.hpp>

namespace sgmcl
{

    template <size_t MAX_DISPARITY>
    inline WinnerTakesAll<MAX_DISPARITY>::WinnerTakesAll(cl_context ctx, cl_device_id device)
        : m_cl_context(ctx), m_cl_device_id(device)
    {
    }

    template <size_t MAX_DISPARITY>
    void WinnerTakesAll<MAX_DISPARITY>::enqueue(DeviceBuffer<uint16_t> &left,
                                                DeviceBuffer<uint16_t> &right,
                                                const DeviceBuffer<uint8_t> &src,
                                                int width,
                                                int height,
                                                int pitch,
                                                float uniqueness,
                                                bool subpixel,
                                                PathType path_type,
                                                cl_command_queue stream)
    {
        if (m_kernel == nullptr)
        {
            std::string kernel_template_types;

            //resource reading
            std::ifstream fileInput1, fileInput2, fileInput3;
            fileInput1.open("data/ocl/inttypes.cl");
            fileInput2.open("data/ocl/utility.cl");
            fileInput3.open("data/ocl/winner_takes_all.cl");

            std::string src1{std::istreambuf_iterator<char>(fileInput1),
                             std::istreambuf_iterator<char>()};
            std::string src2{std::istreambuf_iterator<char>(fileInput2),
                             std::istreambuf_iterator<char>()};
            std::string src3{std::istreambuf_iterator<char>(fileInput3),
                             std::istreambuf_iterator<char>()};
            std::string kernel_src = src1 + src2 + src3;

            //Vertical path aggregation templates
            std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
            int NUM_PATHS = path_type == PathType::SCAN_4PATH ? 4 : 8;
            std::string kernel_NUM_PATHS = "#define NUM_PATHS " + std::to_string(NUM_PATHS) + "\n";
            std::string kernel_COMPUTE_SUBPIXEL = "#define COMPUTE_SUBPIXEL " + std::to_string(subpixel ? 1 : 0) + "\n";
            std::string kernel_WARPS_PER_BLOCK = "#define WARPS_PER_BLOCK " + std::to_string(WARPS_PER_BLOCK) + "\n";
            std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
            std::string kernel_SUBPIXEL_SHIFT = "#define SUBPIXEL_SHIFT " + std::to_string(SubpixelShift()) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
            kernel_src = std::regex_replace(kernel_src, std::regex("@NUM_PATHS@"), kernel_NUM_PATHS);
            kernel_src = std::regex_replace(kernel_src, std::regex("@COMPUTE_SUBPIXEL@"), kernel_COMPUTE_SUBPIXEL);
            kernel_src = std::regex_replace(kernel_src, std::regex("@WARPS_PER_BLOCK@"), kernel_WARPS_PER_BLOCK);
            kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SUBPIXEL_SHIFT@"), kernel_SUBPIXEL_SHIFT);

            m_program.init(m_cl_context, m_cl_device_id, kernel_src);
            //DEBUG
            //std::cout << kernel_src << std::endl;

            m_kernel = m_program.getKernel("winner_takes_all_kernel");
        }

        //setup kernels
        size_t global_size[1] = {
            (height + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK * BLOCK_SIZE};
        size_t local_size[1] = {BLOCK_SIZE};

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &src.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(pitch), &pitch);
        err = clSetKernelArg(m_kernel, 6, sizeof(uniqueness), &uniqueness);

        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel,
                                     1,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");

        //clFinish(stream);
        //cv::Mat debug(height, width, CV_16UC1);
        //clEnqueueReadBuffer(stream, left.data(), true, 0, width * height * 2, debug.data, 0, nullptr, nullptr);
        //cv::imwrite("winn_takes_all.tiff", debug);
        //cv::imshow("winner takes all debug", debug * 2048);
        //cv::waitKey(0);
    }

    template class WinnerTakesAll<64>;
    template class WinnerTakesAll<128>;
    template class WinnerTakesAll<256>;

} // namespace sgmcl
