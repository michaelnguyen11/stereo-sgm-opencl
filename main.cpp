/*
Copyright 2016 fixstars

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

#include <memory>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "src/libsgm_ocl.h"
#include "stereo_camera.hpp"

static bool is_streaming = true;
static void sig_handler(int sig)
{
    is_streaming = false;
}

static cv::CommandLineParser getConfig(int argc, char **argv)
{
    const char *params = "{ help           | false              | print usage          }"
                         "{ fps            | 30                 | (int) Frame rate }"
                         "{ width          | 1280               | (int) Image width }"
                         "{ height         | 720                | (int) Image height }"
                         "{ max_disparity  | 256                | (int) Maximum disparity }"
                         "{ subpixel       | true               | Compute subpixel accuracy }"
                         "{ num_path       | 8                  | (int) Num path to optimize, 4 or 8 }";

    cv::CommandLineParser config(argc, argv, params);
    if (config.get<bool>("help"))
    {
        config.printMessage();
        exit(0);
    }

    return config;
}

void context_error_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data);
std::tuple<cl_context, cl_device_id> initCLCTX(int platform_idx, int device_idx);

int main(int argc, char *argv[])
{
    // handle signal by user
    struct sigaction act;
    act.sa_handler = sig_handler;
    sigaction(SIGINT, &act, NULL);

    // parse config from cmd
    cv::CommandLineParser config = getConfig(argc, argv);

    StereoCameraConfig camConfig;
    camConfig.fps = config.get<int>("fps");
    camConfig.width = config.get<int>("width");
    camConfig.height = config.get<int>("height");

    StereoCamera::Ptr camera = std::make_shared<StereoCamera>(camConfig);
    if (!camera->checkCameraStarted())
    {
        std::cout << "Camera open fail..." << std::endl;
        return 0;
    }

    // Stereo Camera
    cv::Mat frame_0, frame_1, frame_0_rect, frame_1_rect;
    cv::Mat disp16, disp32;
    StereoCameraData camDataCamera;

    cv::FileStorage fs("data/ocams_calibration_720p.xml", cv::FileStorage::READ);

    cv::Mat D_L, K_L, D_R, K_R;
    cv::Mat Rect_L, Proj_L, Rect_R, Proj_R, Q;
    cv::Mat baseline;
    cv::Mat Rotation, Translation;

    fs["D_L"] >> D_L;
    fs["K_L"] >> K_L;
    fs["D_R"] >> D_R;
    fs["K_R"] >> K_R;
    fs["baseline"] >> baseline;
    fs["Rotation"] >> Rotation;
    fs["Translation"] >> Translation;

    // Code to calculate Rotation matrix and Projection matrix for each camera
    cv::Vec3d Translation_2((double *)Translation.data);

    cv::stereoRectify(K_L, D_L, K_R, D_R, cv::Size(camConfig.width, camConfig.height), Rotation, Translation_2,
                      Rect_L, Rect_R, Proj_L, Proj_R, Q, cv::CALIB_ZERO_DISPARITY);

    cv::Mat map11, map12, map21, map22;

    cv::initUndistortRectifyMap(K_L, D_L, Rect_L, Proj_L, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map11, map12);
    cv::initUndistortRectifyMap(K_R, D_R, Rect_R, Proj_R, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map21, map22);

    cl_context cl_ctx;
    cl_device_id cl_device;
    int platform_idx = 0;
    int device_idx = 0;
    std::tie(cl_ctx, cl_device) = initCLCTX(platform_idx, device_idx);
    std::cout << "cl device : " << cl_device << std::endl;
    cl_command_queue cl_queue = clCreateCommandQueue(cl_ctx, cl_device, 0, nullptr);

    sgmcl::Parameters params;
    int disp_size = 256;
    int num_paths = 4;
    params.subpixel = true;

    params.path_type = num_paths == 8 ? sgmcl::PathType::SCAN_8PATH : sgmcl::PathType::SCAN_4PATH;
    params.uniqueness = 0.95f;

    sgmcl::StereoSGM ssgm(camConfig.width,
                            camConfig.height,
                            disp_size,
                            cl_ctx,
                            cl_device,
                            params);

    bool should_close = false;

    cv::Mat disp(camConfig.height, camConfig.width, CV_16UC1);
    cv::Mat disp_color, disp_8u;
    cl_mem d_left, d_right, d_disp;
    d_left = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, camConfig.width * camConfig.height, nullptr, nullptr);
    d_right = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, camConfig.width * camConfig.height, nullptr, nullptr);
    d_disp = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, camConfig.width * camConfig.height * 2, nullptr, nullptr);

    while (is_streaming)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!is_streaming)
        {
            std::cout << "Exit by user signal" << std::endl;
            break;
        }
        if (camera->getCamData(camDataCamera))
        {
            // read the next frames
            frame_0 = camDataCamera.frame_0;
            frame_1 = camDataCamera.frame_1;

            cv::remap(frame_0, frame_0_rect, map11, map12, cv::INTER_LINEAR);
            cv::remap(frame_1, frame_1_rect, map21, map22, cv::INTER_LINEAR);

            clEnqueueWriteBuffer(cl_queue, d_left, true, 0, camConfig.width * camConfig.height, frame_0_rect.data, 0, nullptr, nullptr);
            clEnqueueWriteBuffer(cl_queue, d_right, true, 0, camConfig.width * camConfig.height, frame_1_rect.data, 0, nullptr, nullptr);

            auto t = std::chrono::steady_clock::now();
            //ssgm.execute(left.data, right.data, reinterpret_cast<uint16_t*>(disp.data));
            ssgm.execute(d_left, d_right, d_disp);
            std::chrono::milliseconds dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t);
            clEnqueueReadBuffer(cl_queue, d_disp, true, 0, camConfig.width * camConfig.height * 2, disp.data, 0, nullptr, nullptr);

            cv::Mat disparity_8u, disparity_color;
            disp.convertTo(disparity_8u, CV_8U, 255. / (disp_size * (params.subpixel ? 16 : 1)));
            cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
            const int invalid_disp = static_cast<uint16_t>(ssgm.get_invalid_disparity());

            disparity_color.setTo(cv::Scalar(0, 0, 0), disp == invalid_disp);
            const int64_t fps = 1000 / dur.count();
            cv::putText(disparity_color, "sgm execution time: " + std::to_string(dur.count()) + "[msec] " + std::to_string(fps) + "[FPS]",
                        cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

            cv::imshow("original", frame_0);
            cv::waitKey(1);

            cv::imshow("disp", disparity_color);
            cv::waitKey(1);
        }
    }

    clReleaseMemObject(d_left);
    clReleaseMemObject(d_right);
    clReleaseMemObject(d_disp);

    clReleaseCommandQueue(cl_queue);
    clReleaseDevice(cl_device);
    clReleaseContext(cl_ctx);

    return 0;
}

std::tuple<cl_context, cl_device_id> initCLCTX(int platform_idx, int device_idx)
{
    cl_uint num_platform;
    clGetPlatformIDs(0, nullptr, &num_platform);
    assert((size_t)platform_idx < num_platform);
    std::vector<cl_platform_id> platform_ids(num_platform);
    clGetPlatformIDs(num_platform, platform_ids.data(), nullptr);
    if (platform_ids.size() <= platform_idx)
    {
        std::cout << "Wrong platform index!" << std::endl;
        exit(0);
    }
    cl_uint num_devices;
    clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    assert((size_t)device_idx < num_devices);
    std::vector<cl_device_id> cl_devices(num_devices);
    clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, num_devices, cl_devices.data(), nullptr);
    cl_device_id cl_device = cl_devices[device_idx];
    cl_int err;
    cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_devices[device_idx], context_error_callback, NULL, &err);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error creating context " << err << std::endl;
        throw std::runtime_error("Error creating context!");
    }
    {
        size_t name_size_in_bytes;
        clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_NAME, 0, nullptr, &name_size_in_bytes);
        std::string platform_name;
        platform_name.resize(name_size_in_bytes);
        clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_NAME,
                          platform_name.size(),
                          (void *)platform_name.data(), nullptr);
        std::cout << "Platform name: " << platform_name << std::endl;
    }
    {
        size_t name_size_in_bytes;
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME, 0, nullptr, &name_size_in_bytes);
        std::string dev_name;
        dev_name.resize(name_size_in_bytes);
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME,
                        dev_name.size(),
                        (void *)dev_name.data(), nullptr);
        std::cout << "Device name: " << dev_name << std::endl;
    }
    return std::make_tuple(cl_ctx, cl_device);
}

void context_error_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    std::cout << "opencl error : " << errinfo << std::endl;
}
