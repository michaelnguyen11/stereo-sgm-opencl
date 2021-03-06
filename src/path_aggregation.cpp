#include "path_aggregation.h"

namespace sgmcl
{
    template <size_t MAX_DISPARITY>
    PathAggregation<MAX_DISPARITY>::PathAggregation(cl_context ctx, cl_device_id device)
        : m_cost_buffer(ctx), m_down2up(ctx, device), m_up2down(ctx, device), m_right2left(ctx, device), m_left2right(ctx, device), m_upleft2downright(ctx, device), m_upright2downleft(ctx, device), m_downright2upleft(ctx, device), m_downleft2upright(ctx, device)
    {
        for (size_t i = 0; i < MAX_NUM_PATHS; ++i)
        {
            cl_int err;
            m_streams[i] = clCreateCommandQueue(ctx, device, 0, &err);
            CHECK_OCL_ERROR(err, "Failed to create command queue");
        }
    }

    template <size_t MAX_DISPARITY>
    PathAggregation<MAX_DISPARITY>::~PathAggregation()
    {
        for (int i = 0; i < MAX_NUM_PATHS; ++i)
        {
            if (m_streams[i])
            {
                clReleaseCommandQueue(m_streams[i]);
                m_streams[i] = nullptr;
            }
        }
    }

    template <size_t MAX_DISPARITY>
    const DeviceBuffer<uint8_t> &PathAggregation<MAX_DISPARITY>::get_output() const
    {
        return m_cost_buffer;
    }

    template <size_t MAX_DISPARITY>
    void PathAggregation<MAX_DISPARITY>::enqueue(const DeviceBuffer<uint32_t> &left,
                                                 const DeviceBuffer<uint32_t> &right,
                                                 int width,
                                                 int height,
                                                 PathType path_type,
                                                 unsigned int p1,
                                                 unsigned int p2,
                                                 int min_disp,
                                                 cl_command_queue stream)
    {

        //allocating memory
        const unsigned int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;
        const size_t buffer_size = width * height * MAX_DISPARITY * num_paths;
        const size_t buffer_step = width * height * MAX_DISPARITY;
        if (m_cost_buffer.size() != buffer_size)
        {
            m_cost_buffer.allocate(buffer_size);
            m_sub_buffers.resize(num_paths);
            for (unsigned i = 0; i < num_paths; ++i)
            {
                cl_buffer_region region = {buffer_step * i, buffer_step};
                cl_int err;
                m_sub_buffers[i].setBufferData(nullptr, buffer_step, clCreateSubBuffer(m_cost_buffer.data(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err));
                CHECK_OCL_ERROR(err, "Error creating subbuffer!");
            }
        }

        cl_int err = clFinish(stream);
        CHECK_OCL_ERROR(err, "Error finishing queue");
        m_up2down.enqueue(
            m_sub_buffers[0],
            left,
            right,
            width,
            height,
            p1,
            p2,
            min_disp,
            m_streams[0]);
        m_down2up.enqueue(
            m_sub_buffers[1],
            left,
            right,
            width,
            height,
            p1,
            p2,
            min_disp,
            m_streams[1]);
        m_left2right.enqueue(
            m_sub_buffers[2],
            left,
            right,
            width,
            height,
            p1,
            p2,
            min_disp,
            m_streams[2]);
        m_right2left.enqueue(
            m_sub_buffers[3],
            left,
            right,
            width,
            height,
            p1,
            p2,
            min_disp,
            m_streams[3]);

        if (path_type == PathType::SCAN_8PATH)
        {

            //{
            //    clFinish(stream);
            //    cv::Mat cost(height, width, CV_8UC4);
            //    clEnqueueReadBuffer(stream, right.data(), true, 0, width * height * 4, cost.data, 0, nullptr, nullptr);
            //    cv::imwrite("debug_cost_ocl.tiff", cost);
            //    cv::imshow("cost_res_ocl", cost);
            //    cv::waitKey(0);
            //}

            m_upleft2downright.enqueue(
                m_sub_buffers[4],
                left,
                right,
                width,
                height,
                p1,
                p2,
                min_disp,
                m_streams[4]);
            //{
            //    int path_id = 4;
            //    clFinish(m_streams[path_id]);
            //
            //    cl_buffer_region region = { buffer_step * path_id + width * height * 0, width * height };
            //    cl_mem buff = clCreateSubBuffer(m_cost_buffer.data(),
            //        CL_MEM_READ_WRITE,
            //        CL_BUFFER_CREATE_TYPE_REGION,
            //        &region, &err);
            //
            //
            //    cv::Mat cost(height, width, CV_8UC1);
            //    clEnqueueReadBuffer(stream, buff, true, 0, width * height * 1, cost.data, 0, nullptr, nullptr);
            //    {
            //        std::ofstream mat_txt("debug_cost_ocl.txt");
            //        for (int j = 0; j < cost.rows; ++j)
            //        {
            //            for (int i = 0; i < cost.cols; ++i)
            //            {
            //                mat_txt << static_cast<int>(cost.at<uint8_t>(j, i)) << " ";
            //            }
            //            mat_txt << std::endl;
            //        }
            //        mat_txt.close();
            //    }
            //    //cv::imwrite("debug_cost_ocl.tiff", cost);
            //    cv::imshow("cost_res_ocl", cost);
            //    cv::waitKey(0);
            //}

            m_upright2downleft.enqueue(
                m_sub_buffers[5],
                left,
                right,
                width,
                height,
                p1,
                p2,
                min_disp,
                m_streams[5]);
            m_downright2upleft.enqueue(
                m_sub_buffers[6],
                left,
                right,
                width,
                height,
                p1,
                p2,
                min_disp,
                m_streams[6]);

            m_downleft2upright.enqueue(
                m_sub_buffers[7],
                left,
                right,
                width,
                height,
                p1,
                p2,
                min_disp,
                m_streams[7]);
        }

        for (unsigned i = 0; i < num_paths; ++i)
        {
            cl_int err = clFinish(m_streams[i]);
            CHECK_OCL_ERROR(err, "Error finishing path aggregation!");
        }
    }

    template class PathAggregation<64>;
    template class PathAggregation<128>;
    template class PathAggregation<256>;

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    VerticalPathAggregation<DIRECTION, MAX_DISPARITY>::VerticalPathAggregation(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx), m_cl_device(device)
    {
        m_kernel = nullptr;
    }

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    VerticalPathAggregation<DIRECTION, MAX_DISPARITY>::~VerticalPathAggregation()
    {
        if (m_kernel)
        {
            clReleaseKernel(m_kernel);
            m_kernel = nullptr;
        }
    }

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    void VerticalPathAggregation<DIRECTION, MAX_DISPARITY>::enqueue(DeviceBuffer<uint8_t> &dest,
                                                                    const DeviceBuffer<uint32_t> &left,
                                                                    const DeviceBuffer<uint32_t> &right,
                                                                    int width,
                                                                    int height,
                                                                    unsigned int p1,
                                                                    unsigned int p2,
                                                                    int min_disp,
                                                                    cl_command_queue stream)
    {
        if (!m_kernel)
            init();

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &dest.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(p1), &p1);
        err = clSetKernelArg(m_kernel, 6, sizeof(p2), &p2);
        err = clSetKernelArg(m_kernel, 7, sizeof(min_disp), &min_disp);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

        //setup kernels
        const size_t gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
        const size_t bdim = BLOCK_SIZE;
        //
        size_t global_size[1] = {gdim * bdim};
        size_t local_size[1] = {bdim};
        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel,
                                     1,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error finishing queue");
        //        clFinish(stream);
        //        cv::Mat debug(height, width, CV_8UC4);
        //        clEnqueueReadBuffer(stream, dest.data(), true, 0, width * height * 4, debug.data, 0, nullptr, nullptr);
        //        cv::imshow("vertical path aggregation debug", debug);
        //        cv::waitKey(0);
    }

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    void VerticalPathAggregation<DIRECTION, MAX_DISPARITY>::init()
    {
        if (m_kernel == nullptr)
        {
            std::ifstream fileInput1, fileInput2, fileInput3, fileInput4;
            fileInput1.open("data/ocl/inttypes.cl");
            fileInput2.open("data/ocl/utility.cl");
            fileInput3.open("data/ocl/path_aggregation_common.cl");
            fileInput4.open("data/ocl/path_aggregation_vertical.cl");

            std::string src1{std::istreambuf_iterator<char>(fileInput1),
                             std::istreambuf_iterator<char>()};
            std::string src2{std::istreambuf_iterator<char>(fileInput2),
                             std::istreambuf_iterator<char>()};
            std::string src3{std::istreambuf_iterator<char>(fileInput3),
                             std::istreambuf_iterator<char>()};
            std::string src4{std::istreambuf_iterator<char>(fileInput4),
                             std::istreambuf_iterator<char>()};
            std::string kernel_src = src1 + src2 + src3 + src4;

            //Vertical path aggregation templates
            std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
            std::string kernel_direction = "#define DIRECTION " + std::to_string(DIRECTION) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
            kernel_src = std::regex_replace(kernel_src, std::regex("@DIRECTION@"), kernel_direction);

            static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
            //@DP_BLOCK_SIZE@
            //@SUBGROUP_SIZE@
            //@SIZE@
            //@BLOCK_SIZE@
            //Vertical path aggregation templates
            std::string kernel_DP_BLOCK_SIZE = "#define DP_BLOCK_SIZE " + std::to_string(DP_BLOCK_SIZE) + "\n";
            std::string kernel_SUBGROUP_SIZE = "#define SUBGROUP_SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
            std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
            std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

            //std::cout << "vertical_path_aggregation combined: " << std::endl;
            //std::cout << kernel_src << std::endl;

            m_program.init(m_cl_ctx, m_cl_device, kernel_src);
            //DEBUG
            m_kernel = m_program.getKernel("aggregate_vertical_path_kernel");
        }
    }
    //down2up
    template class VerticalPathAggregation<-1, 64>;
    template class VerticalPathAggregation<-1, 128>;
    template class VerticalPathAggregation<-1, 256>;
    //up2down
    template class VerticalPathAggregation<1, 64>;
    template class VerticalPathAggregation<1, 128>;
    template class VerticalPathAggregation<1, 256>;

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    HorizontalPathAggregation<DIRECTION, MAX_DISPARITY>::HorizontalPathAggregation(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx), m_cl_device(device)
    {
    }

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    HorizontalPathAggregation<DIRECTION, MAX_DISPARITY>::~HorizontalPathAggregation()
    {
        if (m_kernel)
        {
            clReleaseKernel(m_kernel);
            m_kernel = nullptr;
        }
    }

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    void HorizontalPathAggregation<DIRECTION, MAX_DISPARITY>::enqueue(DeviceBuffer<uint8_t> &dest,
                                                                      const DeviceBuffer<uint32_t> &left,
                                                                      const DeviceBuffer<uint32_t> &right,
                                                                      int width,
                                                                      int height,
                                                                      unsigned int p1,
                                                                      unsigned int p2,
                                                                      int min_disp,
                                                                      cl_command_queue stream)
    {
        if (!m_kernel)
            init();

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &dest.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(p1), &p1);
        err = clSetKernelArg(m_kernel, 6, sizeof(p2), &p2);
        err = clSetKernelArg(m_kernel, 7, sizeof(min_disp), &min_disp);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        static const unsigned int PATHS_PER_BLOCK =
            BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

        //setup kernels
        const size_t gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
        const size_t bdim = BLOCK_SIZE;
        //
        size_t global_size[1] = {gdim * bdim};
        size_t local_size[1] = {bdim};
        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel,
                                     1,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        //cl_int errr = clFinish(stream);
        //CHECK_OCL_ERROR(err, "Error finishing queue");
        //cv::Mat debug(height, width, CV_8UC4);
        //clEnqueueReadBuffer(stream, dest.data(), true, 0, width * height * 4, debug.data, 0, nullptr, nullptr);
        //cv::imshow("horizontal path aggregation debug", debug);
        //cv::waitKey(0);
    }

    template <int DIRECTION, unsigned int MAX_DISPARITY>
    void HorizontalPathAggregation<DIRECTION, MAX_DISPARITY>::init()
    {
        if (m_kernel == nullptr)
        {
            //reading cl files
            std::ifstream fileInput1, fileInput2, fileInput3, fileInput4;
            fileInput1.open("data/ocl/inttypes.cl");
            fileInput2.open("data/ocl/utility.cl");
            fileInput3.open("data/ocl/path_aggregation_common.cl");
            fileInput4.open("data/ocl/path_aggregation_horizontal.cl");

            std::string src1{std::istreambuf_iterator<char>(fileInput1),
                             std::istreambuf_iterator<char>()};
            std::string src2{std::istreambuf_iterator<char>(fileInput2),
                             std::istreambuf_iterator<char>()};
            std::string src3{std::istreambuf_iterator<char>(fileInput3),
                             std::istreambuf_iterator<char>()};
            std::string src4{std::istreambuf_iterator<char>(fileInput4),
                             std::istreambuf_iterator<char>()};
            std::string kernel_src = src1 + src2 + src3 + src4;

            //Vertical path aggregation templates
            std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
            std::string kernel_direction = "#define DIRECTION " + std::to_string(DIRECTION) + "\n";
            std::string kernel_DP_BLOCKS_PER_THREAD = "#define DP_BLOCKS_PER_THREAD " + std::to_string(DP_BLOCKS_PER_THREAD) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
            kernel_src = std::regex_replace(kernel_src, std::regex("@DIRECTION@"), kernel_direction);
            kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCKS_PER_THREAD@"), kernel_DP_BLOCKS_PER_THREAD);

            static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
            //@DP_BLOCK_SIZE@
            //@SUBGROUP_SIZE@
            //@SIZE@
            //@BLOCK_SIZE@
            //path aggregation common templates
            std::string kernel_DP_BLOCK_SIZE = "#define DP_BLOCK_SIZE " + std::to_string(DP_BLOCK_SIZE) + "\n";
            std::string kernel_SUBGROUP_SIZE = "#define SUBGROUP_SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
            std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
            std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

            //std::cout << "horizontal_path_aggregation combined: " << std::endl;
            //std::cout << kernel_src << std::endl;

            m_program.init(m_cl_ctx, m_cl_device, kernel_src);
            //DEBUG
            m_kernel = m_program.getKernel("aggregate_horizontal_path_kernel");
        }
    }
    //down2up
    template class HorizontalPathAggregation<-1, 64>;
    template class HorizontalPathAggregation<-1, 128>;
    template class HorizontalPathAggregation<-1, 256>;
    //up2down
    template class HorizontalPathAggregation<1, 64>;
    template class HorizontalPathAggregation<1, 128>;
    template class HorizontalPathAggregation<1, 256>;

    template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
    ObliquePathAggregation<X_DIRECTION, Y_DIRECTION, MAX_DISPARITY>::ObliquePathAggregation(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx), m_cl_device(device)
    {
    }

    template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
    ObliquePathAggregation<X_DIRECTION, Y_DIRECTION, MAX_DISPARITY>::~ObliquePathAggregation()
    {
        if (m_kernel)
        {
            clReleaseKernel(m_kernel);
            m_kernel = nullptr;
        }
    }

    template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
    void ObliquePathAggregation<X_DIRECTION, Y_DIRECTION, MAX_DISPARITY>::enqueue(DeviceBuffer<uint8_t> &dest,
                                                                                  const DeviceBuffer<uint32_t> &left,
                                                                                  const DeviceBuffer<uint32_t> &right,
                                                                                  int width,
                                                                                  int height,
                                                                                  unsigned int p1,
                                                                                  unsigned int p2,
                                                                                  int min_disp,
                                                                                  cl_command_queue stream)
    {
        if (!m_kernel)
            init();

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &dest.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(p1), &p1);
        err = clSetKernelArg(m_kernel, 6, sizeof(p2), &p2);
        err = clSetKernelArg(m_kernel, 7, sizeof(min_disp), &min_disp);

        const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

        const unsigned gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
        const unsigned bdim = BLOCK_SIZE;
        //
        size_t global_size[1] = {gdim * bdim};
        size_t local_size[1] = {bdim};
        err = clEnqueueNDRangeKernel(stream,
                                     m_kernel,
                                     1,
                                     nullptr,
                                     global_size,
                                     local_size,
                                     0, nullptr, nullptr);
        //cl_int errr = clFinish(stream);
        //CHECK_OCL_ERROR(err, "Error finishing queue");
        //cv::Mat debug(height, width, CV_8UC4);
        //clEnqueueReadBuffer(stream, dest.data(), true, 0, width * height * 4, debug.data, 0, nullptr, nullptr);
        //cv::imshow("horizontal path aggregation debug", debug);
        //cv::waitKey(0);
    }

    template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
    void ObliquePathAggregation<X_DIRECTION, Y_DIRECTION, MAX_DISPARITY>::init()
    {
        if (m_kernel == nullptr)
        {
            std::ifstream fileInput1, fileInput2, fileInput3, fileInput4;
            fileInput1.open("data/ocl/inttypes.cl");
            fileInput2.open("data/ocl/utility.cl");
            fileInput3.open("data/ocl/path_aggregation_common.cl");
            fileInput4.open("data/ocl/path_aggregation_oblique.cl");

            std::string src1{std::istreambuf_iterator<char>(fileInput1),
                             std::istreambuf_iterator<char>()};
            std::string src2{std::istreambuf_iterator<char>(fileInput2),
                             std::istreambuf_iterator<char>()};
            std::string src3{std::istreambuf_iterator<char>(fileInput3),
                             std::istreambuf_iterator<char>()};
            std::string src4{std::istreambuf_iterator<char>(fileInput4),
                             std::istreambuf_iterator<char>()};
            std::string kernel_src = src1 + src2 + src3 + src4;

            //Vertical path aggregation templates
            std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
            std::string kernel_x_direction = "#define X_DIRECTION " + std::to_string(X_DIRECTION) + "\n";
            std::string kernel_y_direction = "#define Y_DIRECTION " + std::to_string(Y_DIRECTION) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
            kernel_src = std::regex_replace(kernel_src, std::regex("@X_DIRECTION@"), kernel_x_direction);
            kernel_src = std::regex_replace(kernel_src, std::regex("@Y_DIRECTION@"), kernel_y_direction);

            static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
            //@DP_BLOCK_SIZE@
            //@SUBGROUP_SIZE@
            //@SIZE@
            //@BLOCK_SIZE@
            //path aggregation common templates
            std::string kernel_DP_BLOCK_SIZE = "#define DP_BLOCK_SIZE " + std::to_string(DP_BLOCK_SIZE) + "\n";
            std::string kernel_SUBGROUP_SIZE = "#define SUBGROUP_SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
            std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
            std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
            kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
            kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

            //std::cout << "horizontal_path_aggregation combined: " << std::endl;
            //std::cout << kernel_src << std::endl;

            m_program.init(m_cl_ctx, m_cl_device, kernel_src);
            //DEBUG
            m_kernel = m_program.getKernel("aggregate_oblique_path_kernel");
        }
    }

    //upleft2downright
    template class ObliquePathAggregation<1, 1, 64>;
    template class ObliquePathAggregation<1, 1, 128>;
    template class ObliquePathAggregation<1, 1, 256>;
    //upright2downleft
    template class ObliquePathAggregation<-1, 1, 64>;
    template class ObliquePathAggregation<-1, 1, 128>;
    template class ObliquePathAggregation<-1, 1, 256>;
    //downright2upleft
    template class ObliquePathAggregation<-1, -1, 64>;
    template class ObliquePathAggregation<-1, -1, 128>;
    template class ObliquePathAggregation<-1, -1, 256>;
    //downleft2upright
    template class ObliquePathAggregation<1, -1, 64>;
    template class ObliquePathAggregation<1, -1, 128>;
    template class ObliquePathAggregation<1, -1, 256>;

} // namespace sgmcl
