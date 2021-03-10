#ifndef CL_UTILITIES_H_
#define CL_UTILITIES_H_

#include <string>
#include <iostream>
#include <cstddef>

#include <CL/cl.h>

#define CHECK_OCL_ERROR(err, msg)                                                           \
    if (err != CL_SUCCESS)                                                                  \
    {                                                                                       \
        std::cout << "OCL_ERROR at line " << __LINE__ << ". Message: " << msg << std::endl; \
    }

namespace sgmcl
{
    class DeviceProgram
    {
    public:
        DeviceProgram() = default;
        DeviceProgram(cl_context ctx,
                      cl_device_id device,
                      const std::string &kernel_str)
        {
            init(ctx, device, kernel_str);
        }

        void init(cl_context ctx,
                  cl_device_id device,
                  const std::string &kernel_str)
        {
            if (m_cl_program != nullptr)
            {
                clReleaseProgram(m_cl_program);
                m_cl_program = nullptr;
            }

            cl_int err;
            const char *kernel_src = kernel_str.c_str();
            size_t kenel_src_length = kernel_str.size();
            m_cl_program = clCreateProgramWithSource(ctx, 1, &kernel_src, &kenel_src_length, &err);
            err = clBuildProgram(m_cl_program, 1, &device, nullptr, nullptr, nullptr);

            size_t build_log_size = 0;
            clGetProgramBuildInfo(m_cl_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size);
            std::string build_log;
            build_log.resize(build_log_size);
            clGetProgramBuildInfo(m_cl_program, device, CL_PROGRAM_BUILD_LOG,
                                  build_log_size,
                                  &build_log[0],
                                  nullptr);
            if (build_log.size() > 10)
                std::cout << "OpenCL build info: " << build_log << std::endl;
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Cannot build ocl program!");
            }
        }

        cl_kernel getKernel(const std::string &name)
        {
            cl_int err;
            cl_kernel ret = clCreateKernel(m_cl_program, name.c_str(), &err);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Cannot find ocl kernel: " + name);
            }
            return ret;
        }

        ~DeviceProgram()
        {
            clReleaseProgram(m_cl_program);
        }

        bool isInitialized() const
        {
            return m_cl_program != nullptr;
        }

    private:
        cl_program m_cl_program = nullptr;
    };

    template <typename value_type>
    class DeviceBuffer
    {
    public:
        DeviceBuffer(cl_context ctx = nullptr)
            : m_cl_ctx(ctx), m_data(nullptr), m_size(0), m_owns_data(false)
        {
        }

        explicit DeviceBuffer(cl_context ctx, size_t n)
            : m_cl_ctx(ctx), m_data(nullptr), m_size(0), m_owns_data(false)
        {
            allocate(n);
        }

        explicit DeviceBuffer(cl_context ctx, size_t n, cl_mem data)
            : m_cl_ctx(ctx), m_data(data), m_size(n), m_owns_data(false)
        {
        }

        void setBufferData(cl_context ctx, size_t n, cl_mem data)
        {
            m_cl_ctx = ctx;
            m_size = n;
            m_data = data;
            m_owns_data = false;
        }

        DeviceBuffer(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&obj)
            : m_data(obj.m_data), m_size(obj.m_size)
        {
            obj.m_data = nullptr;
            obj.m_size = 0;
        }

        ~DeviceBuffer()
        {
            destroy();
        }

        void allocate(size_t n)
        {
            if (m_data && m_size >= n)
                return;

            destroy();
            cl_int err;
            m_data = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(value_type) * n, nullptr, &err);
            CHECK_OCL_ERROR(err, "Allocating device buffer");
            m_size = n;
            m_owns_data = true;
        }

        void destroy()
        {
            if (m_data && m_owns_data)
            {
                cl_int err = clReleaseMemObject(m_data);
                CHECK_OCL_ERROR(err, "Destroying device buffer");
                m_data = nullptr;
            }

            m_data = nullptr;
            m_size = 0;
        }

        void fillZero(cl_command_queue queue)
        {
            value_type pattern = 0;
            clEnqueueFillBuffer(queue,
                                m_data,
                                &pattern,
                                sizeof(pattern),
                                0,
                                m_size * sizeof(value_type),
                                0,
                                nullptr,
                                nullptr);
        }

        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer &operator=(DeviceBuffer &&obj)
        {
            m_data = obj.m_data;
            m_size = obj.m_size;
            obj.m_data = nullptr;
            obj.m_size = 0;
            return *this;
        }

        size_t size() const
        {
            return m_size;
        }

        cl_mem &data()
        {
            return m_data;
        }

        const cl_mem &data() const
        {
            return m_data;
        }

    private:
        cl_context m_cl_ctx = nullptr;
        cl_mem m_data = nullptr;
        size_t m_size = 0;
        bool m_owns_data = false;
    };

} // namespace sgmcl

#endif // CL_UTILITIES_H_