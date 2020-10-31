#include "path_aggregation.h"

namespace sgm 
{
namespace cl
{
template<size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::PathAggregation(cl_context ctx, cl_device_id device)
    : m_cost_buffer(ctx)
    , m_down2up(ctx, device)
    , m_up2down(ctx, device)
{
    for (size_t i = 0; i < MAX_NUM_PATHS; ++i)
    {
        cl_int err;
        m_streams[i] = clCreateCommandQueue(ctx, device, 0, &err);
        CHECK_OCL_ERROR(err, "Failed to create command queue");
    }
}

template<size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::~PathAggregation()
{
}

template<size_t MAX_DISPARITY>
const DeviceBuffer<cost_type>& PathAggregation<MAX_DISPARITY>::get_output() const
{
    return m_cost_buffer;
}

template<size_t MAX_DISPARITY>
void PathAggregation<MAX_DISPARITY>::enqueue(const DeviceBuffer<feature_type>& left,
    const DeviceBuffer<feature_type>& right,
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
        for (int i = 0; i < num_paths; ++i)
        {
            cl_buffer_region region = { buffer_step * i, buffer_step };
            cl_int err;
            m_sub_buffers[i].setBufferData(nullptr, buffer_step, clCreateSubBuffer(m_cost_buffer.data(),
                CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region, &err));
            CHECK_OCL_ERROR(err, "Error creating subbuffer!");
        }
    }


    m_up2down.enqueue_aggregate_up2down_path(
        m_sub_buffers[0],
        left,
        right,
        width,
        height,
        p2,
        p2,
        min_disp,
        m_streams[0]
    );
    m_down2up.enqueue_aggregate_up2down_path(
        m_sub_buffers[1],
        left,
        right,
        width,
        height,
        p2,
        p2,
        min_disp,
        m_streams[1]
    );

}

template  PathAggregation<64>;
template  PathAggregation<128>;
template  PathAggregation<256>;


template<int DIRECTION, unsigned int MAX_DISPARITY>
inline void VerticalPathAggregation<DIRECTION, MAX_DISPARITY>::init()
{
    if (m_kernel == nullptr)
    {
        //reading cl files
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_utility = fs.open("src/ocl/utility.cl");
        auto kernel_path_aggregation_common = fs.open("src/ocl/path_aggregation_common.cl");
        auto kernel_path_aggregation_vertical = fs.open("src/ocl/path_aggregation_vertical.cl");

        std::string kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_utility.begin(), kernel_utility.end())
            + std::string(kernel_path_aggregation_common.begin(), kernel_path_aggregation_common.end())
            + std::string(kernel_path_aggregation_vertical.begin(), kernel_path_aggregation_vertical.end());

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
        std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE_P) + "\n";
        std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

        std::cout << "vertical_path_aggregation combined: " << std::endl;
        std::cout << kernel_src << std::endl;

        m_program.init(m_cl_ctx, m_cl_device, kernel_src);
        //DEBUG
        m_kernel = m_program.getKernel("aggregate_vertical_path_kernel");
    }
}
//down2up
template  VerticalPathAggregation<-1, 64>;
template  VerticalPathAggregation<-1, 128>;
template  VerticalPathAggregation<-1, 256>;
//up2down
template  VerticalPathAggregation<1, 64>;
template  VerticalPathAggregation<1, 128>;
template  VerticalPathAggregation<1, 256>;

}
}