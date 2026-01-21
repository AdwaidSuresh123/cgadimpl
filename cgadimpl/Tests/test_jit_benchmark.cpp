#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include "ad/runtime/jit_compiler.hpp"
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n===== JIT PERFORMANCE BENCHMARK (100 STEPS) =====\n";
    std::cout << "Structure based on test_graph_compile.cpp\n\n";

    // ---------- Shapes & Data ----------
    const int B = 32;   
    const int In = 128; 
    const int Out = 64; 
    const int EPOCHS = 100;

    // Use CPU tensors (nova.mse returns host tensor)
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn<float>(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");

    // ---------- Parameters ----------
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn<float>(Shape{{In, Out}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b1");
    
    // Target for MSE Loss
    auto opts_target = TensorOptions().with_req_grad(false);
    auto target = make_tensor(Tensor::randn<float>(Shape{{B, Out}}, opts_target), "target");

    // ---------- Build Computation Graph ----------
    Value Z = matmul(X, W1) + b1;
    Value loss = mse_loss(Z, target);

    // ========== EAGER MODE BENCHMARK ==========
    std::cout << "--- EAGER MODE BENCHMARK (" << EPOCHS << " steps) ---\n";
    
    std::vector<double> eager_times;
    float final_eager_loss = 0.0f;
    
    // Warmup
    for(int i=0; i<5; ++i) {
        ag::zero_grad(loss);
        ag::backward(loss);
    }
    cudaDeviceSynchronize();

    auto eager_start_total = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < EPOCHS; ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        ag::zero_grad(loss); 
        ag::backward(loss);
        cudaDeviceSynchronize();
        
        auto step_end = std::chrono::high_resolution_clock::now();
        eager_times.push_back(std::chrono::duration<double, std::milli>(step_end - step_start).count());
        
        if (i == EPOCHS - 1) {
            final_eager_loss = loss.val().to_cpu().data<float>()[0];
        }
    }
    auto eager_end_total = std::chrono::high_resolution_clock::now();
    double eager_total_time = std::chrono::duration<double, std::milli>(eager_end_total - eager_start_total).count();
    
    // Store eager gradients for verification
    Tensor eager_W1_grad = W1.grad().to_cpu();
    Tensor eager_b1_grad = b1.grad().to_cpu();

    // Stats
    double eager_avg = eager_total_time / EPOCHS;
    std::cout << "Eager Total Time:  " << eager_total_time << " ms\n";
    std::cout << "Eager Avg Time:      " << eager_avg << " ms/step\n";
    std::cout << "Eager Throughput:  " << (1000.0 / eager_avg) << " steps/sec\n\n";

    // ========== JIT COMPILATION ==========
    std::cout << "--- JIT COMPILATION ---\n";
    std::vector<Value> inputs = {X, target};
    std::vector<Value> params = {W1, b1};

    ag::jit::CompileOptions opts;
    opts.include_backward = true;
    
    auto comp_start = std::chrono::high_resolution_clock::now();
    auto comp = ag::jit::compile(loss, inputs, params, opts);
    auto comp_end = std::chrono::high_resolution_clock::now();
    double compilation_time = std::chrono::duration<double, std::milli>(comp_end - comp_start).count();
    std::cout << "Compilation Time:  " << compilation_time << " ms\n\n";

    // ========== JIT MODE BENCHMARK ==========
    std::cout << "--- JIT MODE BENCHMARK (" << EPOCHS << " steps) ---\n";
    
    std::vector<Tensor*> in_ptrs = {&X.node->value, &target.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value};
    
    // Warmup
    for(int i=0; i<5; ++i) {
        std::vector<Tensor> dummy;
        comp.run(in_ptrs, par_ptrs, dummy);
    }
    cudaDeviceSynchronize();

    std::vector<double> jit_times;
    float final_jit_loss = 0.0f;
    std::vector<Tensor> final_jit_outputs;

    auto jit_start_total = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < EPOCHS; ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        std::vector<Tensor> jit_outputs;
        if (!comp.run(in_ptrs, par_ptrs, jit_outputs)) {
            std::cerr << "FAIL: JIT execution failed at step " << i << "\n";
            return 1;
        }
        cudaDeviceSynchronize();
        
        auto step_end = std::chrono::high_resolution_clock::now();
        jit_times.push_back(std::chrono::duration<double, std::milli>(step_end - step_start).count());
        
        if (i == EPOCHS - 1) {
            final_jit_outputs = std::move(jit_outputs);
            final_jit_loss = final_jit_outputs[0].to_cpu().data<float>()[0];
        }
    }
    auto jit_end_total = std::chrono::high_resolution_clock::now();
    double jit_total_time = std::chrono::duration<double, std::milli>(jit_end_total - jit_start_total).count();

    // Stats
    double jit_avg = jit_total_time / EPOCHS;
    std::cout << "JIT Total Time:    " << jit_total_time << " ms\n";
    std::cout << "JIT Avg Time:      " << jit_avg << " ms/step\n";
    std::cout << "JIT Throughput:    " << (1000.0 / jit_avg) << " steps/sec\n\n";

    // ========== VERIFICATION & PRECISION ==========
    std::cout << "--- VERIFICATION & ARITHMETIC PRECISION ---\n";
    float loss_diff = std::abs(final_eager_loss - final_jit_loss);
    std::cout << "Final Eager Loss:    " << final_eager_loss << "\n";
    std::cout << "Final JIT Loss:      " << final_jit_loss << "\n";
    std::cout << "Loss Difference:     " << loss_diff << "\n";

    // Gradient MAD
    Tensor jit_W1_grad = final_jit_outputs[1].to_cpu();
    Tensor jit_b1_grad = final_jit_outputs[2].to_cpu();

    float w1_mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_W1_grad - jit_W1_grad, ag::current_stream())).data<float>()[0];
    float b1_mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_b1_grad - jit_b1_grad, ag::current_stream())).data<float>()[0];

    std::cout << "W1 Grad MAD:         " << w1_mad << "\n";
    std::cout << "b1 Grad MAD:         " << b1_mad << "\n\n";

    // ========== FINAL COMPARISON TABLE ==========
    std::cout << "======================================================================\n";
    std::cout << "                      FINAL COMPARISON (100 STEPS)\n";
    std::cout << "======================================================================\n";
    std::cout << "Metric                 │ Eager Mode      │ JIT Mode        │ Speedup\n";
    std::cout << "-----------------------┼-----------------┼-----------------┼----------\n";
    std::cout << "Total Time (ms)        │ " << std::setw(15) << eager_total_time << " │ " << std::setw(15) << jit_total_time << " │ " << (eager_total_time / jit_total_time) << "x\n";
    std::cout << "Avg Time (ms)          │ " << std::setw(15) << eager_avg        << " │ " << std::setw(15) << jit_avg        << " │ " << (eager_avg / jit_avg) << "x\n";
    std::cout << "Throughput (step/s)    │ " << std::setw(15) << (1000.0 / eager_avg) << " │ " << std::setw(15) << (1000.0 / jit_avg) << " │ " << (jit_avg / eager_avg) << "x\n";
    std::cout << "-----------------------┴-----------------┴-----------------┴----------\n";
    std::cout << "Compilation Overhead:  " << compilation_time << " ms\n";
    std::cout << "Break-even Point:      ~" << (int)(compilation_time / (std::max(1e-6, eager_avg - jit_avg))) << " steps\n";
    std::cout << "======================================================================\n";

    if (loss_diff < 1e-4 && w1_mad < 1e-4 && b1_mad < 1e-4) {
        std::cout << "✅ VERIFICATION PASSED: JIT is numerically accurate.\n";
        return 0;
    } else {
        std::cout << "❌ VERIFICATION FAILED: Numerical discrepancy detected.\n";
        return 1;
    }
}
