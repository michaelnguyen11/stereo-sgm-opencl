#pragma once
#include <CL/cl.h>
#include <inttypes.h>

#include "common.h"
#include <memory>

namespace sgmcl
{
    struct CudaStereoSGMResources;

    class StereoSGM
    {
    public:
        /**
            * @param width Processed image's width.
            * @param height Processed image's height.
            * @param disparity_size It must be 64, 128 or 256.
            * @param input_depth_bits Processed image's bits per pixel. It must be 8 or 16.
            * @param inout_type Specify input/output pointer type. See sgm::EXECUTE_TYPE.
            * @attention
            */
        StereoSGM(int width,
                  int height,
                  int disparity_size,
                  cl_context ctx,
                  cl_device_id cl_device,
                  Parameters param);

        ~StereoSGM();

        /**
            * Execute stereo semi global matching.
            * @param left_pixels  A pointer stored input left image in device memory.
            * @param right_pixels A pointer stored input right image in device memory.
            * @param dst          Output pointer in device memory. User must allocate enough memory.
            * @attention
            * You need to allocate dst memory at least width x height x sizeof(element_type) bytes.
            * The element_type is uint8_t for output_depth_bits == 8 and uint16_t for output_depth_bits == 16.
            * Note that dst element value would be multiplied StereoSGM::SUBPIXEL_SCALE if subpixel option was enabled.
            * Value of Invalid disparity is equal to return value of `get_invalid_disparity` member function.
            */
        void execute(cl_mem left_pixels, cl_mem right_pixels, cl_mem dst);

        /**
            * Generate invalid disparity value from Parameter::min_disp and Parameter::subpixel
            * @attention
            * Cast properly if you receive disparity value as `unsigned` type.
            * See sample/movie for an example of this.
            */
        int get_invalid_disparity() const;

    private:
        StereoSGM(const StereoSGM &) = delete;
        StereoSGM &operator=(const StereoSGM &) = delete;

        std::unique_ptr<CudaStereoSGMResources> m_cu_res;

        int m_width = -1;
        int m_height = -1;
        int m_max_disparity = -1;

        Parameters m_params;

        //cl context info
        cl_context m_cl_ctx;
        cl_device_id m_cl_device;
        cl_command_queue m_cl_cmd_queue;
    };
} // namespace sgmcl
