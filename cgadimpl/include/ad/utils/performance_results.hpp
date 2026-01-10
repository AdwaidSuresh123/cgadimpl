#ifndef PERFORMANCE_RESULTS_HPP
#define PERFORMANCE_RESULTS_HPP

#include "performance_utils.hpp"
#include <fstream>
#include <vector>
#include <map>

namespace ag {
namespace utils {

class ResultsWriter {
public:
    static void write_csv(const std::string& filename, const std::map<std::string, PerformanceMetrics>& results) {
        std::ofstream file(filename);
        file << "Operation,Forward_ms,Backward_ms,Total_ms,CPU_Peak_MB,GPU_Used_MB,Throughput_ops_s,Bandwidth_GB_s,GPU_Temp_C,GPU_Util_%,Accuracy_Error\n";
        for (auto const& [name, m] : results) {
            file << name << ","
                 << m.forward_time_ms << ","
                 << m.backward_time_ms << ","
                 << m.total_latency_ms << ","
                 << m.memory.cpu_peak_kb / 1024.0 << ","
                 << m.memory.gpu_used_bytes / (1024.0 * 1024.0) << ","
                 << m.throughput << ","
                 << m.bandwidth_gb_s << ","
                 << m.gpu_temp_c << ","
                 << m.gpu_utilization << ","
                 << m.accuracy_error << "\n";
        }
    }

    static void write_json(const std::string& filename, const std::map<std::string, PerformanceMetrics>& results) {
        std::ofstream file(filename);
        file << "{\n";
        bool first = true;
        for (auto const& [name, m] : results) {
            if (!first) file << ",\n";
            file << "  \"" << name << "\": {\n"
                 << "    \"forward_ms\": " << m.forward_time_ms << ",\n"
                 << "    \"backward_ms\": " << m.backward_time_ms << ",\n"
                 << "    \"total_ms\": " << m.total_latency_ms << ",\n"
                 << "    \"cpu_peak_mb\": " << m.memory.cpu_peak_kb / 1024.0 << ",\n"
                 << "    \"gpu_used_mb\": " << m.memory.gpu_used_bytes / (1024.0 * 1024.0) << ",\n"
                 << "    \"throughput\": " << m.throughput << ",\n"
                 << "    \"bandwidth_gb_s\": " << m.bandwidth_gb_s << ",\n"
                 << "    \"gpu_temp_c\": " << m.gpu_temp_c << ",\n"
                 << "    \"gpu_utilization\": " << m.gpu_utilization << ",\n"
                 << "    \"accuracy_error\": " << m.accuracy_error << "\n"
                 << "  }";
            first = false;
        }
        file << "\n}\n";
    }

    static void print_summary(const std::map<std::string, PerformanceMetrics>& results) {
        std::cout << "\n" << std::setw(20) << std::left << "Operation" 
                  << " | " << std::setw(10) << "Fwd (ms)"
                  << " | " << std::setw(10) << "Bwd (ms)"
                  << " | " << std::setw(10) << "GPU (MB)"
                  << " | " << std::setw(10) << "Error" << "\n";
        std::cout << std::string(70, '-') << "\n";
        for (auto const& [name, m] : results) {
            std::cout << std::setw(20) << std::left << name 
                      << " | " << std::setw(10) << std::fixed << std::setprecision(3) << m.forward_time_ms
                      << " | " << std::setw(10) << m.backward_time_ms
                      << " | " << std::setw(10) << m.memory.gpu_used_bytes / (1024.0 * 1024.0)
                      << " | " << std::setw(10) << std::scientific << m.accuracy_error << "\n";
        }
    }
};

} // namespace utils
} // namespace ag

#endif // PERFORMANCE_RESULTS_HPP
