#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include "ad/runtime/jit_compiler.hpp"

using namespace ag;

int main() {
    std::cout << "===== JIT COMPILER TEST WITH GPU METRICS =====\n";

    // ---------- GPU Hardware Info ----------
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU Device: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "-------------------------------------------\n";
    }

    // ---------- Shapes & Data ----------
    const int B = 8;   // batch size
    const int In = 16; // input dim
    const int Out = 10; // output dim

    auto opts_const = TensorOptions().with_device(Device::CUDA);
    Tensor Xt = Tensor::randn<float>(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");

    auto opts_param = TensorOptions().with_req_grad(true).with_device(Device::CUDA);
    auto W1 = make_tensor(Tensor::randn<float>(Shape{{In, Out}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b1");
    
    auto opts_target = TensorOptions().with_req_grad(false).with_device(Device::CUDA);
    auto target = make_tensor(Tensor::randn<float>(Shape{{B, Out}}, opts_target), "target");

    // ---------- Forward Pass ----------
    Value Z = matmul(X, W1) + b1;
    Value loss = mse_loss(Z, target);
    
    std::cout << "Eager forward pass completed.\n";
    
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << std::fixed << std::setprecision(6) << eager_loss << "\n";

    // ---------- JIT Compilation ----------
    std::cout << "\nCompiling graph...\n";
    std::vector<Value> inputs = {X, target};
    std::vector<Value> params = {W1, b1};

    ag::jit::CompileOptions opts;
    opts.include_backward = true;
    auto comp = ag::jit::compile(loss, inputs, params, opts);

    std::cout << "Graph compilation successful.\n";

    // ---------- JIT Execution with Timing ----------
    std::cout << "\nRunning compiled graph...\n";

    std::vector<Tensor*> in_ptrs = {&X.node->value, &target.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value};

    // CUDA Timing Setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<Tensor> jit_outputs;
    
    cudaEventRecord(start);
    bool ok = comp.run(in_ptrs, par_ptrs, jit_outputs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (!ok) {
        std::cerr << "FAIL: JIT execution failed.\n";
        return 1;
    }

    std::cout << "Compiled execution successful.\n";
    std::cout << "JIT Execution Time: " << std::fixed << std::setprecision(4) << milliseconds << " ms\n";
    
    // Calculate GFLOPS (Very approximate for this small graph)
    // Matmul 1: 8x16 @ 16x10 = 2 * 8 * 16 * 10 = 2560 ops
    // Add bias: 8 * 10 = 80 ops
    // MSE Loss: ~ 3 * 8 * 10 = 240 ops
    // Backward matmul: 16x8 @ 8x10 = 2 * 16 * 8 * 10 = 2560 ops
    // Total approx: 5500 ops
    double total_ops = 5500.0;
    double gflops = (total_ops / (milliseconds / 1000.0)) / 1e9;
    std::cout << "JIT Throughput (Approx): " << gflops << " GFLOPS\n";
    
    // ---------- Verification ----------
    float compiled_loss = jit_outputs[0].to_cpu().data<float>()[0];

    std::cout << "\n--- Verification ---\n";
    std::cout << "Eager Loss:    " << eager_loss << "\n";
    std::cout << "Compiled Loss: " << compiled_loss << "\n";

    if (std::abs(eager_loss - compiled_loss) < 1e-4f) {
        std::cout << " PASS: Loss matches.\n";
    } else {
        std::cout << "❌ FAIL: Loss mismatch.\n";
    }
    
    bool grads_match = true;
    
    // W1 grad
    {
        Tensor eager_grad = W1.grad().to_cpu();
        Tensor jit_grad = jit_outputs[1].to_cpu();
        float mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad - jit_grad, ag::current_stream())).data<float>()[0];
        std::cout << "W1 Grad MAD: " << std::scientific << mad << "\n";
        if (mad > 1e-4f) grads_match = false;
    }
    
    // b1 grad
    {
        Tensor eager_grad = b1.grad().to_cpu();
        Tensor jit_grad = jit_outputs[2].to_cpu();
        float mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad - jit_grad, ag::current_stream())).data<float>()[0];
        std::cout << "b1 Grad MAD: " << std::scientific << mad << "\n";
        if (mad > 1e-4f) grads_match = false;
    }
    
    if (grads_match) {
        std::cout << "PASS: Gradients match.\n";
    } else {
        std::cout << "FAIL: Gradient mismatch.\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}