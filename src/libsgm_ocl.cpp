#include "libsgm_ocl.h"
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include "sgm.h"
#include "device_buffer.h"
#include "sgm_details.h"
#include <memory>

namespace sgmcl
{
    class SemiGlobalMatchingBase
    {
    public:
        virtual void execute(DeviceBuffer<uint16_t> &dst_L,
                             DeviceBuffer<uint16_t> &dst_R,
                             const DeviceBuffer<uint8_t> &src_L,
                             const DeviceBuffer<uint8_t> &src_R,
                             DeviceBuffer<uint32_t> &feature_buff_l,
                             DeviceBuffer<uint32_t> &feature_buff_r,
                             int w,
                             int h,
                             int sp,
                             int dp,
                             Parameters &param,
                             cl_command_queue queue) = 0;

        virtual ~SemiGlobalMatchingBase() {}
    };

    template <int DISP_SIZE>
    class SemiGlobalMatchingImpl : public SemiGlobalMatchingBase
    {
    public:
        SemiGlobalMatchingImpl(cl_context ctx, cl_device_id device)
            : sgm_engine_(ctx, device) {}
        void execute(
            DeviceBuffer<uint16_t> &dst_L,
            DeviceBuffer<uint16_t> &dst_R,
            const DeviceBuffer<uint8_t> &src_L,
            const DeviceBuffer<uint8_t> &src_R,
            DeviceBuffer<uint32_t> &feature_buff_l,
            DeviceBuffer<uint32_t> &feature_buff_r,
            int w,
            int h,
            int sp,
            int dp,
            Parameters &param,
            cl_command_queue queue) override
        {
            sgm_engine_.enqueue(dst_L,
                                dst_R,
                                src_L,
                                src_R,
                                feature_buff_l,
                                feature_buff_r,
                                w,
                                h,
                                sp,
                                dp,
                                param,
                                queue);
        }
        virtual ~SemiGlobalMatchingImpl() {}

    private:
        SemiGlobalMatching<DISP_SIZE> sgm_engine_;
    };

    struct CudaStereoSGMResources
    {
        DeviceBuffer<uint32_t> d_feature_buffer_left;
        DeviceBuffer<uint32_t> d_feature_buffer_right;
        DeviceBuffer<uint8_t> d_src_left;
        DeviceBuffer<uint8_t> d_src_right;
        DeviceBuffer<uint16_t> d_left_disp;
        DeviceBuffer<uint16_t> d_right_disp;
        DeviceBuffer<uint16_t> d_tmp_left_disp;
        DeviceBuffer<uint16_t> d_tmp_right_disp;
        DeviceBuffer<uint8_t> d_u8_out_disp;

        std::unique_ptr<SemiGlobalMatchingBase> sgm_engine;
        SGMDetails sgm_details;

        CudaStereoSGMResources(int width_,
                               int height_,
                               int disparity_size_,
                               int src_pitch_,
                               int dst_pitch_,
                               cl_context ctx,
                               cl_device_id device,
                               cl_command_queue queue)
            : d_feature_buffer_left(ctx), d_feature_buffer_right(ctx), d_src_left(ctx), d_src_right(ctx), d_left_disp(ctx), d_right_disp(ctx), d_tmp_left_disp(ctx), d_tmp_right_disp(ctx), d_u8_out_disp(ctx), sgm_details(ctx, device)
        {
            if (disparity_size_ == 64)
                sgm_engine = std::make_unique<SemiGlobalMatchingImpl<64>>(ctx, device);
            else if (disparity_size_ == 128)
                sgm_engine = std::make_unique<SemiGlobalMatchingImpl<128>>(ctx, device);
            else if (disparity_size_ == 256)
                sgm_engine = std::make_unique<SemiGlobalMatchingImpl<256>>(ctx, device);
            else
                throw std::logic_error("depth bits must be 8 or 16, and disparity size must be 64 or 128");

            d_feature_buffer_left.allocate(static_cast<size_t>(width_ * height_));
            d_feature_buffer_right.allocate(static_cast<size_t>(width_ * height_));

            this->d_left_disp.allocate(dst_pitch_ * height_);
            this->d_right_disp.allocate(dst_pitch_ * height_);

            this->d_tmp_left_disp.allocate(dst_pitch_ * height_);
            this->d_tmp_right_disp.allocate(dst_pitch_ * height_);

            this->d_left_disp.fillZero(queue);
            this->d_right_disp.fillZero(queue);
            this->d_tmp_left_disp.fillZero(queue);
            this->d_tmp_right_disp.fillZero(queue);
        }

        ~CudaStereoSGMResources()
        {
            sgm_engine.reset();
        }
    };

    StereoSGM::StereoSGM(int width,
                         int height,
                         int disparity_size,
                         cl_context ctx,
                         cl_device_id cl_device,
                         Parameters param)
        : m_width(width), m_height(height), m_max_disparity(disparity_size), m_cl_ctx(ctx), m_cl_device(cl_device), m_params(param)
    {
        //create command queue

        cl_int err;
        m_cl_cmd_queue = clCreateCommandQueue(m_cl_ctx, m_cl_device, 0, &err);
        CHECK_OCL_ERROR(err, "Failed to create command queue");

        // check values
        if (disparity_size != 64 && disparity_size != 128 && disparity_size != 256)
        {
            throw std::logic_error("disparity size must be 64, 128 or 256");
        }
        if (param.path_type != PathType::SCAN_4PATH && param.path_type != PathType::SCAN_8PATH)
        {
            throw std::logic_error("Path type must be PathType::SCAN_4PATH or PathType::SCAN_8PATH");
        }

        m_cu_res = std::make_unique<CudaStereoSGMResources>(width,
                                                            height,
                                                            disparity_size,
                                                            width,
                                                            width,
                                                            m_cl_ctx,
                                                            m_cl_device,
                                                            m_cl_cmd_queue);
    }

    StereoSGM::~StereoSGM()
    {
        m_cu_res.reset();
    }

    void StereoSGM::execute(cl_mem left_pixels, cl_mem right_pixels, cl_mem dst)
    {
        DeviceBuffer<uint8_t> left_img(m_cl_ctx,
                                       m_width * m_height * sizeof(uint8_t),
                                       left_pixels);
        DeviceBuffer<uint8_t> right_img(m_cl_ctx,
                                        m_width * m_height * sizeof(uint8_t),
                                        right_pixels);
        DeviceBuffer<uint16_t> out_disp(m_cl_ctx,
                                        m_width * m_height * sizeof(uint16_t),
                                        dst);

        m_cu_res->sgm_engine->execute(m_cu_res->d_tmp_left_disp,
                                      m_cu_res->d_tmp_right_disp,
                                      left_img,
                                      right_img,
                                      m_cu_res->d_feature_buffer_left,
                                      m_cu_res->d_feature_buffer_right,
                                      m_width,
                                      m_height,
                                      m_width,
                                      m_width,
                                      m_params,
                                      m_cl_cmd_queue);

        m_cu_res->sgm_details.median_filter(m_cu_res->d_tmp_left_disp,
                                            out_disp,
                                            m_width,
                                            m_height,
                                            m_width,
                                            m_cl_cmd_queue);

        m_cu_res->sgm_details.median_filter(m_cu_res->d_tmp_right_disp,
                                            m_cu_res->d_right_disp,
                                            m_width,
                                            m_height,
                                            m_width,
                                            m_cl_cmd_queue);

        m_cu_res->sgm_details.check_consistency(out_disp,
                                                m_cu_res->d_right_disp,
                                                left_img,
                                                m_width,
                                                m_height,
                                                m_width,
                                                m_width,
                                                m_params.subpixel,
                                                m_params.LR_max_diff,
                                                m_cl_cmd_queue);

        m_cu_res->sgm_details.correct_disparity_range(out_disp,
                                                      m_width,
                                                      m_height,
                                                      m_width,
                                                      m_params.subpixel,
                                                      m_params.min_disp,
                                                      m_cl_cmd_queue);
        clFinish(m_cl_cmd_queue);
    }

    int StereoSGM::get_invalid_disparity() const
    {
        return (m_params.min_disp - 1) * (m_params.subpixel ? SubpixelScale() : 1);
    }
} // namespace sgmcl
