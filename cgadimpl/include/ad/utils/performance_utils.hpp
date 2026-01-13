#ifndef PERFORMANCE_UTILS_HPP
#define PERFORMANCE_UTILS_HPP

#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

namespace ag {
namespace utils {

struct MemoryStats {
    size_t cpu_peak_kb = 0;
    size_t cpu_rss_kb = 0;
    size_t gpu_used_bytes = 0;
    size_t gpu_total_bytes = 0;
};

struct PerformanceMetrics {
    double forward_time_ms = 0;
    double backward_time_ms = 0;
    double total_latency_ms = 0;
    MemoryStats memory;
    double throughput = 0; // ops/sec
    double bandwidth_gb_s = 0;
    float gpu_temp_c = 0;
    float gpu_utilization = 0;
    double accuracy_error = 0;
};

class HighResTimer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

class CUDAEventTimer {
public:
    CUDAEventTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CUDAEventTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }

    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }

private:
    cudaEvent_t start_event, stop_event;
};

inline MemoryStats get_memory_stats() {
    MemoryStats stats;
    
    // CPU Memory via /proc/self/status
    std::ifstream status_file("/proc/self/status");
    std::string line;
    if (status_file.is_open()) {
        while (std::getline(status_file, line)) {
            if (line.find("VmPeak:") == 0) {
                std::stringstream ss(line.substr(7));
                ss >> stats.cpu_peak_kb;
            } else if (line.find("VmRSS:") == 0) {
                std::stringstream ss(line.substr(6));
                ss >> stats.cpu_rss_kb;
            }
        }
    }

    // GPU Memory via CUDA
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        stats.gpu_used_bytes = total_bytes - free_bytes;
        stats.gpu_total_bytes = total_bytes;
    }

    return stats;
}

inline float get_gpu_temperature() {
    FILE* pipe = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", "r");
    if (!pipe) return 0.0f;
    char buffer[128];
    float temp = 0.0f;
    if (fgets(buffer, 128, pipe) != NULL) {
        temp = atof(buffer);
    }
    pclose(pipe);
    return temp;
}

inline float get_gpu_utilization() {
    FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", "r");
    if (!pipe) return 0.0f;
    char buffer[128];
    float util = 0.0f;
    if (fgets(buffer, 128, pipe) != NULL) {
        util = atof(buffer);
    }
    pclose(pipe);
    return util;
}

} // namespace utils
} // namespace ag

#endif // PERFORMANCE_UTILS_HPP
