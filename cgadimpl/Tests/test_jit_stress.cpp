#include <iostream>
#include <iomanip>
#include "ad/runtime/jit_compiler.hpp"
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    // Stress Chain: Add -> Matmul -> Gelu -> Softmax
    const int N = 64;
    auto opts = TensorOptions().with_device(Device::CUDA);  // GPU tensors
    
    Tensor A_t = Tensor::randn<float>(Shape{{N, N}}, opts) * 0.1f;
    Tensor B_t = Tensor::randn<float>(Shape{{N, N}}, opts) * 0.1f;
    Value A = make_tensor(A_t, "A");
    Value B = make_tensor(B_t, "B");
    
    // Chain: sum = A + B
    Value sum = A + B;
    // prod = sum @ A
    Value prod = matmul(sum, A);
    // gelu = gelu(prod)
    Value post_gelu = gelu(prod);
    // soft = softmax(gelu, dim=1)
    // Value result = softmax_row(post_gelu);
    Value result = post_gelu;
    Value final_val = result;
    
    // Eager execution
    ag::backward(final_val);
    float eager_loss = final_val.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss = " << eager_loss << "\n";
    
    // JIT Compilation
    std::cout << "\nCompiling graph...\n";
    ag::jit::CompileOptions compile_opts; 
    compile_opts.include_backward = false;
    
    std::vector<Value> inputs = {A, B};
    std::vector<Value> params = {};
    
    auto comp = ag::jit::compile(final_val, inputs, params, compile_opts);
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
    Tensor eager_mat = final_val.val().to_cpu();
    Tensor jit_mat = outputs[0].to_cpu();
    
    std::cout << "\n--- Verification ---\n";
    Tensor diff = OwnTensor::abs(eager_mat - jit_mat, (cudaStream_t)ag::current_stream());
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
