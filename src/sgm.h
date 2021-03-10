/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
#include <memory>
#include <cstdint>
#include "device_buffer.h"
#include "common.h"

namespace sgmcl
{

    template <size_t MAX_DISPARITY>
    class SemiGlobalMatching
    {
    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;

    public:
        SemiGlobalMatching(cl_context context, cl_device_id device);
        ~SemiGlobalMatching();

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
            cl_command_queue stream);
    };
} // namespace sgmcl
