#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <memory>

#include <CL/cl.h>

#include "common.h"
#include "cl_utilities.h"

#include "census_transform.h"
#include "path_aggregation.h"
#include "winner_takes_all.h"
#include "sgm_details.h"

namespace sgmcl
{
    class SemiGlobalMatchingBase
    {
    public:
        virtual void execute(DeviceBuffer<uint16_t> &dest_left,
                             DeviceBuffer<uint16_t> &dest_right,
                             const DeviceBuffer<uint8_t> &src_left,
                             const DeviceBuffer<uint8_t> &src_right,
                             DeviceBuffer<uint32_t> &feature_buffer_left,
                             DeviceBuffer<uint32_t> &feature_buffer_right,
                             int width,
                             int height,
                             int src_pitch,
                             int dst_pitch,
                             const Parameters &param,
                             cl_command_queue queue) = 0;

        virtual ~SemiGlobalMatchingBase() {}
    };

    template <size_t MAX_DISPARITY>
    class SemiGlobalMatching : public SemiGlobalMatchingBase
    {
    public:
        SemiGlobalMatching(cl_context ctx, cl_device_id device);
        virtual ~SemiGlobalMatching() {}

        void execute(DeviceBuffer<uint16_t> &dest_left,
                     DeviceBuffer<uint16_t> &dest_right,
                     const DeviceBuffer<uint8_t> &src_left,
                     const DeviceBuffer<uint8_t> &src_right,
                     DeviceBuffer<uint32_t> &feature_buffer_left,
                     DeviceBuffer<uint32_t> &feature_buffer_right,
                     int width,
                     int height,
                     int src_pitch,
                     int dst_pitch,
                     const Parameters &param,
                     cl_command_queue queue) override;

    private:
        CensusTransform m_census;
        PathAggregation<MAX_DISPARITY> m_path_aggregation;
        WinnerTakesAll<MAX_DISPARITY> m_winner_takes_all;
    };

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

        int m_width;
        int m_height;

        Parameters m_params;

        cl_context m_cl_ctx;
        cl_device_id m_cl_device;
        cl_command_queue m_cl_cmd_queue;

        std::unique_ptr<SemiGlobalMatchingBase> sgm_engine;
        SGMDetails sgm_details;

        DeviceBuffer<uint32_t> d_feature_buffer_left;
        DeviceBuffer<uint32_t> d_feature_buffer_right;
        DeviceBuffer<uint8_t> d_src_left;
        DeviceBuffer<uint8_t> d_src_right;
        DeviceBuffer<uint16_t> d_left_disp;
        DeviceBuffer<uint16_t> d_right_disp;
        DeviceBuffer<uint16_t> d_tmp_left_disp;
        DeviceBuffer<uint16_t> d_tmp_right_disp;
        DeviceBuffer<uint8_t> d_u8_out_disp;
    };
} // namespace sgmcl
