#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

#include "ad/ag_all.hpp"
#include "ad/runtime/jit_compiler.hpp"
#include <cuda_runtime.h>
#include "TensorLib.h"

using namespace ag;
using namespace OwnTensor;

void print_test_header(const std::string& name) {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << " TEST: " << name << "\n";
    std::cout << std::string(50, '=') << "\n";
}

bool verify_results(const std::string& name, float eager_loss, float jit_loss, 
                   const std::vector<std::string>& param_names,
                   const std::vector<Value>& param_values,
                   const std::vector<Tensor>& jit_outputs) {
    std::cout << "\n--- Verification: " << name << " ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Eager Loss:    " << eager_loss << "\n";
    std::cout << "Compiled Loss: " << jit_loss << "\n";

    bool pass = true;
    if (std::abs(eager_loss - jit_loss) > 1e-4f) {
        std::cout << "FAIL: Loss mismatch!\n";
        pass = false;
    } else {
        std::cout << "PASS: Loss matches.\n";
    }

    for (size_t i = 0; i < param_values.size(); ++i) {
        Tensor eager_grad = param_values[i].grad().to_cpu();
        Tensor jit_grad = jit_outputs[i + 1].to_cpu(); // index 0 is loss

        float mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad - jit_grad, ag::current_stream())).data<float>()[0];
        std::cout << param_names[i] << " Grad MAD: " << mad;
        if (mad < 1e-4f) {
            std::cout << " [OK]\n";
        } else {
            std::cout << " [FAIL]\n";
            pass = false;
        }
    }
    return pass;
}

void test_large_scale_mlp() {
    print_test_header("Large Scale MLP (128x512 -> 1024 -> 512 -> 10)");
    
    const int B = 128;
    const int In = 512;
    const int H1 = 1024;
    const int H2 = 512;
    const int Out = 10;

    Value X = make_tensor(Tensor::randn<float>(Shape{{B, In}}, TensorOptions().with_device(Device::CUDA)), "X");
    Value Y = make_tensor(Tensor::randn<float>(Shape{{B, Out}}, TensorOptions().with_device(Device::CUDA)), "Y");

    auto opts = TensorOptions().with_req_grad(true).with_device(Device::CUDA);
    auto W1 = make_tensor(Tensor::randn<float>(Shape{{In, H1}}, opts) * 0.01f, "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, H1}}, opts), "b1");
    auto W2 = make_tensor(Tensor::randn<float>(Shape{{H1, H2}}, opts) * 0.01f, "W2");
    auto b2 = make_tensor(Tensor::zeros(Shape{{1, H2}}, opts), "b2");
    auto W3 = make_tensor(Tensor::randn<float>(Shape{{H2, Out}}, opts) * 0.01f, "W3");
    auto b3 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts), "b3");

    // Forward
    Value L1 = relu(matmul(X, W1) + b1);
    Value L2 = relu(matmul(L1, W2) + b2);
    Value logits = matmul(L2, W3) + b3;
    Value loss = mse_loss(logits, Y);

    // Eager
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];

    // JIT
    std::vector<Value> inputs = {X, Y};
    std::vector<Value> params = {W1, b1, W2, b2, W3, b3};
    ag::jit::CompileOptions c_opts;
    c_opts.include_backward = true;
    auto comp = ag::jit::compile(loss, inputs, params, c_opts);
    
    std::vector<Tensor*> in_t = {&X.node->value, &Y.node->value};
    std::vector<Tensor*> pa_t = {&W1.node->value, &b1.node->value, &W2.node->value, &b2.node->value, &W3.node->value, &b3.node->value};
    std::vector<Tensor> jit_outs;
    comp.run(in_t, pa_t, jit_outs);

    verify_results("Large Scale MLP", eager_loss, jit_outs[0].to_cpu().data<float>()[0], 
                  {"W1", "b1", "W2", "b2", "W3", "b3"}, params, jit_outs);

    // ---------- Performance Metrics ----------
    std::cout << "\n--- Performance Analysis: Large MLP ---\n";
    
    ag::jit::JITMetrics metrics = comp.getMetrics();
    std::cout << "Total FLOPS (estimated):      " << metrics.total_flops << "\n";
    std::cout << "IO Bytes (transferred):       " << metrics.io_bytes << "\n";
    
    double ai = (double)metrics.total_flops / std::max((int64_t)1, metrics.io_bytes);
    std::cout << "Arithmetic Intensity:         " << ai << " FLOP/byte\n";

    const int iterations = 100; // Stable count
    auto start_eager = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Value L1_e = relu(matmul(X, W1) + b1);
        Value L2_e = relu(matmul(L1_e, W2) + b2);
        Value logits_e = matmul(L2_e, W3) + b3;
        Value loss_e = mse_loss(logits_e, Y);
        ag::backward(loss_e);
    }
    cudaStreamSynchronize(ag::current_stream());
    auto end_eager = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eager_dur = (end_eager - start_eager) / iterations;
    
    std::vector<Tensor*> in_t_ptr = {&X.node->value, &Y.node->value};
    std::vector<Tensor*> pa_t_ptr = {&W1.node->value, &b1.node->value, &W2.node->value, &b2.node->value, &W3.node->value, &b3.node->value};
    std::vector<Tensor> dummy_outs;

    auto start_jit = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!comp.run(in_t_ptr, pa_t_ptr, dummy_outs)) {
             std::cerr << "FAIL: JIT execution failed in timing loop\n";
             return;
        }
        dummy_outs.clear();
    }
    cudaStreamSynchronize(ag::current_stream());
    auto end_jit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> jit_dur = (end_jit - start_jit) / iterations;

    double gflops = (metrics.total_flops * 1e-9) / jit_dur.count();
    double speedup = eager_dur.count() / jit_dur.count();
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Eager Avg Time:      " << eager_dur.count() * 1000.0 << " ms\n";
    std::cout << "Compiled Avg Time:   " << jit_dur.count() * 1000.0 << " ms\n";
    std::cout << "Speedup Ratio:       " << speedup << "x\n";
    std::cout << "JIT Throughput:      " << gflops << " GFLOPS\n";
}

void test_complex_broadcasting() {
    print_test_header("Complex Broadcasting: (A[8,16] + B[1,16]) * C[8,1]");
    
    Value A = make_tensor(Tensor::randn<float>(Shape{{8, 16}}, TensorOptions()), "A");
    Value B = make_tensor(Tensor::randn<float>(Shape{{1, 16}}, TensorOptions().with_req_grad(true)), "B");
    Value C = make_tensor(Tensor::randn<float>(Shape{{8, 1}}, TensorOptions().with_req_grad(true)), "C");

    Value out = (A + B) * C;
    Value loss = mean_all(out);

    // Eager
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];

    // JIT
    std::vector<Value> inputs = {A};
    std::vector<Value> params = {B, C};
    ag::jit::CompileOptions c_opts;
    c_opts.include_backward = true;
    auto comp = ag::jit::compile(loss, inputs, params, c_opts);
    
    std::vector<Tensor*> in_t = {&A.node->value};
    std::vector<Tensor*> pa_t = {&B.node->value, &C.node->value};
    std::vector<Tensor> jit_outs;
    if (!comp.run(in_t, pa_t, jit_outs)) {
        std::cerr << "FAIL: JIT execution failed for Complex Broadcasting\n";
        return;
    }

    verify_results("Broadcasting", eager_loss, jit_outs[0].to_cpu().data<float>()[0], 
                  {"B", "C"}, params, jit_outs);
}

void test_reduction_chains() {
    print_test_header("Reduction Chains: MeanAll(RowSum(X[8,16]))");
    
    Value X = make_tensor(Tensor::randn<float>(Shape{{8, 16}}, TensorOptions().with_req_grad(true)), "X");
    Value loss = mean_all(rowsum(X));

    // Eager
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];

    // JIT
    std::vector<Value> inputs = {};
    std::vector<Value> params = {X};
    ag::jit::CompileOptions c_opts;
    c_opts.include_backward = true;
    auto comp = ag::jit::compile(loss, inputs, params, c_opts);
    
    std::vector<Tensor*> in_t = {};
    std::vector<Tensor*> pa_t = {&X.node->value};
    std::vector<Tensor> jit_outs;
    comp.run(in_t, pa_t, jit_outs);

    verify_results("Reduction Chains", eager_loss, jit_outs[0].to_cpu().data<float>()[0], 
                  {"X"}, params, jit_outs);
}

void test_non_contiguous_inputs() {
    print_test_header("Non-Contiguous Inputs: JIT(X.transpose())");
    
    Tensor base = Tensor::randn<float>(Shape{{16, 8}}, TensorOptions());
    Tensor transposed = base.transpose(-2, -1); // Shape [8, 16], non-contiguous
    
    Value X = make_tensor(transposed, "X");
    Value W = make_tensor(Tensor::randn<float>(Shape{{16, 32}}, TensorOptions().with_req_grad(true)), "W");
    Value out = matmul(X, W);
    Value loss = mean_all(out);

    // Eager
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];

    // JIT
    std::vector<Value> inputs = {X};
    std::vector<Value> params = {W};
    ag::jit::CompileOptions c_opts;
    c_opts.include_backward = true;
    auto comp = ag::jit::compile(loss, inputs, params, c_opts);
    
    std::vector<Tensor*> in_t = {&X.node->value};
    std::vector<Tensor*> pa_t = {&W.node->value};
    std::vector<Tensor> jit_outs;
    comp.run(in_t, pa_t, jit_outs);

    verify_results("Non-Contiguous", eager_loss, jit_outs[0].to_cpu().data<float>()[0], 
                  {"W"}, params, jit_outs);
}

int main() {
    try {
        test_large_scale_mlp();
        test_complex_broadcasting();
        test_reduction_chains();
        test_non_contiguous_inputs();
        
        std::cout << "\nALL JIT EDGE CASE TESTS FINISHED.\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
