#include "sgm.h"
#include "census_transform.h"
#include "path_aggregation.h"
#include "winner_takes_all.h"

namespace sgmcl
{
    template <size_t MAX_DISPARITY>
    class SemiGlobalMatching<MAX_DISPARITY>::Impl
    {

    private:
        CensusTransform m_census;
        PathAggregation<MAX_DISPARITY> m_path_aggregation;
        WinnerTakesAll<MAX_DISPARITY> m_winner_takes_all;

    public:
        Impl(cl_context ctx, cl_device_id device)
            : m_census(ctx, device), m_path_aggregation(ctx, device), m_winner_takes_all(ctx, device)
        {
        }

        void enqueue(
            DeviceBuffer<uint16_t> &dest_left,
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
            cl_command_queue stream)
        {
            m_census.enqueue(src_left, feature_buffer_left, width, height, src_pitch, stream);
            m_census.enqueue(src_right, feature_buffer_right, width, height, src_pitch, stream);
            m_path_aggregation.enqueue(
                feature_buffer_left,
                feature_buffer_right,
                width, height,
                param.path_type,
                param.P1,
                param.P2,
                param.min_disp,
                stream);
            m_winner_takes_all.enqueue(
                dest_left, dest_right,
                m_path_aggregation.get_output(),
                width, height, dst_pitch,
                param.uniqueness, param.subpixel, param.path_type,
                stream);
        }
    };

    template <size_t MAX_DISPARITY>
    inline SemiGlobalMatching<MAX_DISPARITY>::SemiGlobalMatching(cl_context context, cl_device_id device)
        : m_impl(std::make_unique<Impl>(context, device))
    {
    }

    template <size_t MAX_DISPARITY>
    SemiGlobalMatching<MAX_DISPARITY>::~SemiGlobalMatching()
    {
    }

    template <size_t MAX_DISPARITY>
    void SemiGlobalMatching<MAX_DISPARITY>::enqueue(
        DeviceBuffer<uint16_t> &dest_left,
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
        cl_command_queue stream)
    {
        m_impl->enqueue(
            dest_left,
            dest_right,
            src_left,
            src_right,
            feature_buffer_left,
            feature_buffer_right,
            width, height,
            src_pitch, dst_pitch,
            param,
            stream);
    }

    template class SemiGlobalMatching<64>;
    template class SemiGlobalMatching<128>;
    template class SemiGlobalMatching<256>;
} // namespace sgmcl
