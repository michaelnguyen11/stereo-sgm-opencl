#include "libsgm_ocl.h"

namespace sgmcl
{
    template <size_t MAX_DISPARITY>
    SemiGlobalMatching<MAX_DISPARITY>::SemiGlobalMatching(cl_context ctx, cl_device_id device)
        : m_census(ctx, device), m_path_aggregation(ctx, device), m_winner_takes_all(ctx, device)
    {
    }

    template <size_t MAX_DISPARITY>
    void SemiGlobalMatching<MAX_DISPARITY>::execute(DeviceBuffer<uint16_t> &dest_left,
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
                                                    cl_command_queue queue)
    {
        m_census.enqueue(src_left, feature_buffer_left, width, height, src_pitch, queue);
        m_census.enqueue(src_right, feature_buffer_right, width, height, src_pitch, queue);
        m_path_aggregation.enqueue(feature_buffer_left,
                                   feature_buffer_right,
                                   width, height,
                                   param.path_type,
                                   param.P1,
                                   param.P2,
                                   param.min_disp,
                                   queue);
        m_winner_takes_all.enqueue(dest_left, dest_right,
                                   m_path_aggregation.get_output(),
                                   width, height, dst_pitch,
                                   param.uniqueness, param.subpixel, param.path_type,
                                   queue);
    }

    template class SemiGlobalMatching<64>;
    template class SemiGlobalMatching<128>;
    template class SemiGlobalMatching<256>;

    StereoSGM::StereoSGM(int width,
                         int height,
                         int disparity_size,
                         cl_context ctx,
                         cl_device_id cl_device,
                         Parameters param)
        : m_width(width), m_height(height),
          m_cl_ctx(ctx), m_cl_device(cl_device), m_params(param),
          d_feature_buffer_left(ctx), d_feature_buffer_right(ctx),
          d_src_left(ctx), d_src_right(ctx),
          d_left_disp(ctx), d_right_disp(ctx),
          d_tmp_left_disp(ctx), d_tmp_right_disp(ctx),
          d_u8_out_disp(ctx), sgm_details(ctx, cl_device)
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

        if (disparity_size == 64)
            sgm_engine = std::make_unique<SemiGlobalMatching<64>>(ctx, cl_device);
        else if (disparity_size == 128)
            sgm_engine = std::make_unique<SemiGlobalMatching<128>>(ctx, cl_device);
        else if (disparity_size == 256)
            sgm_engine = std::make_unique<SemiGlobalMatching<256>>(ctx, cl_device);
        else
            throw std::logic_error("depth bits must be 8 or 16, and disparity size must be 64 or 128");

        d_feature_buffer_left.allocate(static_cast<size_t>(m_width * m_height));
        d_feature_buffer_right.allocate(static_cast<size_t>(m_width * m_height));

        d_left_disp.allocate(m_width * m_height);
        d_right_disp.allocate(m_width * m_height);
        d_tmp_left_disp.allocate(m_width * m_height);
        d_tmp_right_disp.allocate(m_width * m_height);

        d_left_disp.fillZero(m_cl_cmd_queue);
        d_right_disp.fillZero(m_cl_cmd_queue);
        d_tmp_left_disp.fillZero(m_cl_cmd_queue);
        d_tmp_right_disp.fillZero(m_cl_cmd_queue);
    }

    StereoSGM::~StereoSGM()
    {
        sgm_engine.reset();
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

        sgm_engine->execute(d_tmp_left_disp,
                            d_tmp_right_disp,
                            left_img,
                            right_img,
                            d_feature_buffer_left,
                            d_feature_buffer_right,
                            m_width,
                            m_height,
                            m_width,
                            m_width,
                            m_params,
                            m_cl_cmd_queue);

        sgm_details.median_filter(d_tmp_left_disp,
                                  out_disp,
                                  m_width,
                                  m_height,
                                  m_width,
                                  m_cl_cmd_queue);

        sgm_details.median_filter(d_tmp_right_disp,
                                  d_right_disp,
                                  m_width,
                                  m_height,
                                  m_width,
                                  m_cl_cmd_queue);

        sgm_details.check_consistency(out_disp,
                                      d_right_disp,
                                      left_img,
                                      m_width,
                                      m_height,
                                      m_width,
                                      m_width,
                                      m_params.subpixel,
                                      m_params.LR_max_diff,
                                      m_cl_cmd_queue);

        sgm_details.correct_disparity_range(out_disp,
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
