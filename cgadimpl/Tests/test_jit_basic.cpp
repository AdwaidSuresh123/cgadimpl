#include <iostream>
#include <iomanip>
#include "ad/runtime/jit_compiler.hpp"
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    // Simple test: Add and Matmul
    const int N = 16;
    auto opts = TensorOptions().with_device(Device::CUDA);  // GPU tensors
    
    Tensor A_t = Tensor::randn<float>(Shape{{N, N}}, opts);
    Tensor B_t = Tensor::randn<float>(Shape{{N, N}}, opts);
    
    Value A = make_tensor(A_t, "A");
    Value B = make_tensor(B_t, "B");
    
    // Graph: A @ B
    Value prod = matmul(A, B);
    Value loss = prod; // Just return the product
    
    // Eager execution
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss = " << eager_loss << "\n";
    
    // JIT Compilation
    std::cout << "\nCompiling graph...\n";
    ag::jit::CompileOptions compile_opts;
    compile_opts.include_backward = false;
    
    std::vector<Value> inputs = {A, B};
    std::vector<Value> params = {};
    
    auto comp = ag::jit::compile(loss, inputs, params, compile_opts);
    std::cout << "Graph compilation successful.\n";
    
    // Run compiled function
    std::cout << "\nRunning compiled graph...\n";
    std::vector<Tensor*> in_ptrs = {&A.node->value, &B.node->value};
    std::vector<Tensor*> par_ptrs = {};
    std::vector<Tensor> outputs;
    
    if (!comp.run(in_ptrs, par_ptrs, outputs)) {
        std::cerr << "FAIL: JIT execution failed.\n";
        return 1;
    }
    
    std::cout << "Compiled execution successful.\n";
    
    // Verification
    Tensor jit_loss = outputs[0];
    
    std::cout << "\n--- Verification ---\n";
    Tensor eager_val = loss.val().to_cpu();
    Tensor jit_val = jit_loss.to_cpu();
    
    // For matrix match, we can check Mean Absolute Difference
    Tensor diff = OwnTensor::abs(eager_val - jit_val, (cudaStream_t)ag::current_stream());
    float mad = OwnTensor::reduce_mean(diff).data<float>()[0];
    std::cout << "Mean Absolute Difference: " << mad << "\n";
    
    if (mad < 1e-3f) {
        std::cout << "✅ PASS: Output matches.\n";
        return 0;
    } else {
        std::cerr << "❌ FAIL: Output mismatch.\n";
        return 1;
    }
}
