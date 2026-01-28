#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include "ad/runtime/jit_compiler.hpp"

using namespace ag;

int main() {
    std::cout << "===== JIT COMPILER TEST =====\n";

    // ---------- Shapes & Data ----------
    const int B = 8;   // batch size
    const int In = 16; // input dim
    const int Out = 10; // output dim

    // Use default options (CPU, requires_grad=false)
    auto opts_const = TensorOptions().with_device(Device::CUDA);
    Tensor Xt = Tensor::randn<float>(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");

    // ---------- Parameters ----------
    // Parameters must have requires_grad=true
    auto opts_param = TensorOptions().with_req_grad(true).with_device(Device::CUDA);
    auto W1 = make_tensor(Tensor::randn<float>(Shape{{In, Out}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b1");
    
    // Target for MSE Loss
    auto opts_target = TensorOptions().with_req_grad(false).with_device(Device::CUDA);
    auto target = make_tensor(Tensor::randn<float>(Shape{{B, Out}}, opts_target), "target");

    // ---------- Forward Pass (using only JIT-supported ops) ----------
    // A simple linear layer: Z = X @ W1 + b1
    Value Z = matmul(X, W1) + b1;

    // MSE Loss
    Value loss = mse_loss(Z, target);
    
    std::cout << "Eager forward pass completed.\n";
    
    // ---------- Eager Backward ----------
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << eager_loss << "\n";

    // ---------- JIT Compilation ----------
    std::cout << "\nCompiling graph...\n";

    // Tell the compiler which leaves are runtime inputs vs. trainable parameters
    std::vector<Value> inputs = {X, target};
    std::vector<Value> params = {W1, b1};

    // The 'loss' Value is the root of the graph to be compiled
    ag::jit::CompileOptions opts;
    opts.include_backward = true;
    auto comp = ag::jit::compile(loss, inputs, params, opts);

    std::cout << "Graph compilation successful.\n";

    // ---------- JIT Execution ----------
    std::cout << "\nRunning compiled graph...\n";

    // Prepare raw tensor pointers for the run() method
    std::vector<Tensor*> in_ptrs = {&X.node->value, &target.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value};

    std::vector<Tensor> jit_outputs;
    bool ok = comp.run(in_ptrs, par_ptrs, jit_outputs);
    
    if (!ok) {
        std::cerr << "FAIL: JIT execution failed (shape guard or other error).\n";
        return 1;
    }

    std::cout << "Compiled execution successful.\n";
    
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
    
    // Verify Gradients
    // jit_outputs[1] -> grad W1
    // jit_outputs[2] -> grad b1
    bool grads_match = true;
    
    // W1 grad
    {
        Tensor eager_grad = W1.grad().to_cpu();
        Tensor jit_grad = jit_outputs[1].to_cpu();
        float eager_mean = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad, ag::current_stream())).data<float>()[0];
        float jit_mean = OwnTensor::reduce_mean(OwnTensor::abs(jit_grad, ag::current_stream())).data<float>()[0];
        float mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad - jit_grad, ag::current_stream())).data<float>()[0];
        
        std::cout << "W1 Grad Eager Mean: " << eager_mean << "\n";
        std::cout << "W1 Grad JIT Mean:   " << jit_mean << "\n";
        std::cout << "W1 Grad MAD: " << mad << "\n";
        if (mad > 1e-4f) grads_match = false;
    }
    
    // b1 grad
    {
        Tensor eager_grad = b1.grad().to_cpu();
        Tensor jit_grad = jit_outputs[2].contiguous().to_cpu();
        float eager_mean = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad, ag::current_stream())).data<float>()[0];
        float jit_mean = OwnTensor::reduce_mean(OwnTensor::abs(jit_grad, ag::current_stream())).data<float>()[0];
        float mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad - jit_grad, ag::current_stream())).data<float>()[0];
        
        std::cout << "b1 Grad Eager Mean: " << eager_mean << "\n";
        std::cout << "b1 Grad JIT Mean:   " << jit_mean << "\n";
        std::cout << "b1 Grad MAD: " << mad << "\n";
        if (mad > 1e-4f) grads_match = false;
    }
    
    // ---------- Performance Metrics ----------
    std::cout << "\n--- Performance Analysis ---\n";
    
    ag::jit::JITMetrics metrics = comp.getMetrics();
    std::cout << "Total FLOPS (estimated):      " << metrics.total_flops << "\n";
    std::cout << "IO Bytes (transferred):       " << metrics.io_bytes << "\n";
    std::cout << "Intermediate Memory (bytes):  " << metrics.total_intermediate_bytes << "\n";
    
    double ai = (double)metrics.total_flops / std::max((int64_t)1, metrics.io_bytes);
    std::cout << "Arithmetic Intensity:         " << ai << " FLOP/byte\n";

    // Timing Eager (100 iterations)
    const int iterations = 3;
    auto start_eager = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Value Z_e = matmul(X, W1) + b1;
        Value loss_e = mse_loss(Z_e, target);
        ag::backward(loss_e);
    }
    auto end_eager = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eager_dur = (end_eager - start_eager) / iterations;
    
    // Timing JIT (100 iterations)
    auto start_jit = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        comp.run(in_ptrs, par_ptrs, jit_outputs);
    }
    auto end_jit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> jit_dur = (end_jit - start_jit) / iterations;

    double gflops = (metrics.total_flops * 1e-9) / jit_dur.count();
    double speedup = eager_dur.count() / jit_dur.count();
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Eager Avg Time:      " << eager_dur.count() * 1000.0 << " ms\n";
    std::cout << "Compiled Avg Time:   " << jit_dur.count() * 1000.0 << " ms\n";
    std::cout << "Speedup Ratio:       " << speedup << "x\n";
    std::cout << "JIT Throughput:      " << gflops << " GFLOPS\n";

    // Roofline Analysis (Simplified)
    // Assuming a generic modern GPU peak (e.g. RTX 3090 ~35 TFLOPS, 936 GB/s => Balance ~37 FLOP/B)
    // Or just report if AI is low or high.
    std::cout << "Roofline Analysis:   ";
    if (ai < 10.0) {
        std::cout << "Memory Bound (Low Arithmetic Intensity)\n";
    } else {
        std::cout << "Compute Bound (High Arithmetic Intensity)\n";
    }

    return 0;
}