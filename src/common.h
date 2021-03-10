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

#ifndef SGM_TYPES_HPP
#define SGM_TYPES_HPP

#include <cstdint>
#include <regex>
#include <vector>
#include <fstream>

#include "cl_utilities.h"

namespace sgmcl
{
    /**
         Indicates number of scanlines which will be used.
        */
    enum class PathType
    {
        SCAN_4PATH, //>! Horizontal and vertical paths.
        SCAN_8PATH  //>! Horizontal, vertical and oblique paths.
    };

    struct Parameters
    {
        // Penalty on the disparity change by plus or minus 1 between nieghbor pixels.
        int P1 = 10;
        // Penalty on the disparity change by more than 1 between neighbor pixels.
        int P2 = 120;
        // Margin in ratio by which the best cost function value should be at least second one.
        float uniqueness = 0.95f;
        // Disparity value has 4 fractional bits if subpixel option is enabled.
        bool subpixel = false;
        // Number of scanlines used in cost aggregation.
        PathType path_type = PathType::SCAN_8PATH;
        // Minimum possible disparity value.
        int min_disp = 0;
        // Acceptable difference pixels which is used in LR check consistency. LR check consistency will be disabled if this value is set to negative.
        int LR_max_diff = 1;
    };

    inline constexpr int SubpixelShift()
    {
        return 4;
    }

    inline constexpr int SubpixelScale()
    {
        return (1 << SubpixelShift());
    }
} // namespace sgmcl

#endif
