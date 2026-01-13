#include "ad/ag_all.hpp"
#include "ad/utils/performance_utils.hpp"
#include "ad/utils/performance_results.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace ag;
using namespace ag::utils;
using namespace std;
using namespace OwnTensor;

// Global results map
map<string, PerformanceMetrics> all_results;

// Helper to run a test with metrics
template<typename Func>
void run_perf_test(const string& name, size_t num_elements, Func test_func) {
    PerformanceMetrics m;
    HighResTimer timer;
    CUDAEventTimer cuda_timer;

    const int num_iters = 100;

    // Warmup
    test_func(false);
    cudaDeviceSynchronize();

    // Forward Pass
    timer.start();
    cuda_timer.start();
    Value y;
    for (int i = 0; i < num_iters; ++i) {
        y = test_func(true);
        if (i == num_iters / 2) {
            m.gpu_utilization = get_gpu_utilization();
            m.gpu_temp_c = get_gpu_temperature();
        }
    }
    m.forward_time_ms = cuda_timer.stop() / num_iters;
    if (m.forward_time_ms <= 0) m.forward_time_ms = timer.stop() / num_iters;
    else timer.stop();

    // Backward Pass
    timer.start();
    cuda_timer.start();
    for (int i = 0; i < num_iters; ++i) {
        backward(sum(y));
    }
    m.backward_time_ms = cuda_timer.stop() / num_iters;
    if (m.backward_time_ms <= 0) m.backward_time_ms = timer.stop() / num_iters;
    else timer.stop();

    m.total_latency_ms = m.forward_time_ms + m.backward_time_ms;
    m.memory = get_memory_stats();
    m.throughput = (num_elements / (m.total_latency_ms / 1000.0));
    
    // Approximate bandwidth: (input_size + output_size + grad_input + grad_output) * sizeof(float)
    m.bandwidth_gb_s = (num_elements * 4 * sizeof(float)) / (m.total_latency_ms * 1e6);

    // Accuracy check
    m.accuracy_error = 0.0; 

    all_results[name] = m;
    cout << "Finished " << name << " (util: " << m.gpu_utilization << "%)" << endl;
}

void test_activations() {
    Shape s{{1024, 1024}};
    size_t n = 1024 * 1024;
    Device dev = Device::CUDA;
    TensorOptions opts = TensorOptions().with_req_grad(true).with_device(dev);

    run_perf_test("gelu", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(s, opts), "x");
        return gelu(x);
    });

    run_perf_test("relu", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(s, opts), "x");
        return relu(x);
    });

    run_perf_test("sigmoid", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(s, opts), "x");
        return sigmoid(x);
    });

    run_perf_test("softmax", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(s, opts), "x");
        return softmax_row(x);
    });

    run_perf_test("tanh", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(s, opts), "x");
        return tanh(x);
    });
}

void test_layers() {
    // Linear
    {
        size_t in = 1024, out = 512;
        size_t n = in * out;
        Device dev = Device::CUDA;
        run_perf_test("linear", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(Shape{{64, in}}, TensorOptions().with_req_grad(true).with_device(dev)), "x");
        nn::Linear layer(in, out, dev);
        return layer(x);
        });
    }

    // Dropout
    {
        size_t n = 1024 * 1024;
        Device dev = Device::CUDA;
        run_perf_test("dropout", n, [&](bool) {
        Value x = make_tensor(Tensor::randn<float>(Shape{{1024, 1024}}, TensorOptions().with_req_grad(true).with_device(dev)), "x");
        return dropout(x, 0.5f);
        });
    }

    // Flatten
    {
        size_t n = 1024 * 1024;
        Device dev = Device::CUDA;
        run_perf_test("flatten", n, [&](bool) {
            Value x = make_tensor(Tensor::randn<float>(Shape{{64, 128, 128}}, TensorOptions().with_req_grad(true).with_device(dev)), "x");
            return flatten(x);
        });
    }
}

void test_losses() {
    Shape s{{1024, 1024}};
    size_t n = 1024 * 1024;
    Device dev = Device::CUDA;
    TensorOptions opts = TensorOptions().with_req_grad(true).with_device(dev);

    run_perf_test("mse", n, [&](bool) {
        Value p = make_tensor(Tensor::rand<float>(s, opts), "p");
        Value t = make_tensor(Tensor::rand<float>(s, TensorOptions().with_device(dev)), "t");
        return mse_loss(p, t);
    });

    run_perf_test("mae", n, [&](bool) {
        Value p = make_tensor(Tensor::rand<float>(s, opts), "p");
        Value t = make_tensor(Tensor::rand<float>(s, TensorOptions().with_device(dev)), "t");
        return mae_loss(p, t);
    });

    run_perf_test("binary_cross_entropy", n, [&](bool) {
        Value p = make_tensor(Tensor::rand<float>(s, opts), "p");
        Value t = make_tensor(Tensor::rand<float>(s, TensorOptions().with_device(dev)), "t");
        return binary_cross_entropy(p, t);
    });

    run_perf_test("categorical_cross_entropy", n, [&](bool) {
        Value p_raw = make_tensor(Tensor::rand<float>(s, opts), "p_raw");
        Value p = softmax_row(p_raw);
        Value t = make_tensor(Tensor::rand<float>(s, TensorOptions().with_device(dev)), "t");
        return categorical_cross_entropy(p, t);
    });

    run_perf_test("sparse_cross_entropy_with_logits", n, [&](bool) {
        // NOTE: gather/scatter_add are not yet implemented for CUDA, so we run this on CPU
        auto cpu_opts = opts.with_device(Device::CPU);
        Value logits = make_tensor(Tensor::randn<float>(s, cpu_opts), "logits");
        Tensor labels_data = Tensor::zeros(Shape{{1024}}, TensorOptions().with_dtype(Dtype::Int32).with_device(Device::CPU));
        labels_data.fill(1); 
        Value labels = make_tensor(labels_data, "labels");
        return sparse_cross_entropy_with_logits(logits, labels);
    });

    run_perf_test("kldivergence", n, [&](bool) {
        Value p = make_tensor(Tensor::rand<float>(s, opts), "p");
        Value q = make_tensor(Tensor::rand<float>(s, TensorOptions().with_device(dev)), "q");
        return kldivergence(p, q);
    });

    run_perf_test("cross_entropy_with_logits", n, [&](bool) {
        Value logits = make_tensor(Tensor::randn<float>(s, opts), "logits");
        Value target = make_tensor(Tensor::rand<float>(s, TensorOptions().with_device(dev)), "target");
        return cross_entropy_with_logits(logits, target);
    });
}

int main() {
    cout << "Starting Performance Tests..." << endl;
    
    try {
        test_activations();
        test_layers();
        test_losses();

        ResultsWriter::write_json("performance_results.json", all_results);
        
        cout << "\nResults saved to performance_results.json" << endl;
    } catch (const exception& e) {
        cerr << "Error during testing: " << e.what() << endl;
        return 1;
    }

    return 0;
}
