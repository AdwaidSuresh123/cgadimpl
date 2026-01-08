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
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");

    // ---------- Parameters ----------
    // Parameters must have requires_grad=true
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn(Shape{{In, Out}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b1");

    // ---------- Forward Pass (using only JIT-supported ops) ----------
    // A simple linear layer: Z = X @ W1 + b1
    auto eager_start = std::chrono::high_resolution_clock::now();
    
    Value Z = matmul(X, W1) + b1;

    // To create a scalar loss, we can sum all elements of Z
    Value loss = sum(Z);
    
    auto eager_end = std::chrono::high_resolution_clock::now();
    auto eager_duration = std::chrono::duration_cast<std::chrono::microseconds>(eager_end - eager_start).count();
    
    std::cout << "Eager forward pass completed.\n";
    debug::print_value("Eager Loss", loss);
    std::cout << "⏱️  Eager Execution Time: " << eager_duration << " μs (" << eager_duration / 1000.0 << " ms)\n";

    // ---------- JIT Compilation ----------
    std::cout << "\nCompiling graph...\n";

    auto compile_start = std::chrono::high_resolution_clock::now();
    
    // Tell the compiler which leaves are runtime inputs vs. trainable parameters
    std::vector<Value> inputs = {X};
    std::vector<Value> params = {W1, b1};

    // The 'loss' Value is the root of the graph to be compiled
    auto comp = ag::jit::compile(loss, inputs, params);

    auto compile_end = std::chrono::high_resolution_clock::now();
    auto compile_duration = std::chrono::duration_cast<std::chrono::microseconds>(compile_end - compile_start).count();

    std::cout << "Graph compilation successful.\n";
    std::cout << "⏱️  JIT Compilation Time: " << compile_duration << " μs (" << compile_duration / 1000.0 << " ms)\n";

    // ---------- JIT Execution ----------
    std::cout << "\nRunning compiled graph...\n";

    auto exec_start = std::chrono::high_resolution_clock::now();
    
    // Prepare raw tensor pointers for the run() method
    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value};

    Tensor compiled_out; // This will receive the output
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    auto exec_end = std::chrono::high_resolution_clock::now();
    auto exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(exec_end - exec_start).count();
    
    if (!ok) {
        std::cerr << "FAIL: JIT execution failed (shape guard or other error).\n";
        return 1;
    }

    std::cout << "Compiled execution successful.\n";
    debug::print_tensor("Compiled Loss", compiled_out);
    std::cout << "⏱️  Compiled Execution Time: " << exec_duration << " μs (" << exec_duration / 1000.0 << " ms)\n";

    // ---------- Verification ----------
    // Compare the scalar result from the eager pass and the compiled pass
    float eager_val = loss.val().to_cpu().data<float>()[0];
    float compiled_val = compiled_out.to_cpu().data<float>()[0];

    std::cout << "\n--- Verification ---\n";
    std::cout << "Eager Result:    " << eager_val << "\n";
    std::cout << "Compiled Result: " << compiled_val << "\n";

    if (std::abs(eager_val - compiled_val) < 1e-5f) {
        std::cout << "✅ PASS: Eager and compiled results match.\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }

    // Performance Summary
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "Eager Execution:     " << eager_duration / 1000.0 << " ms\n";
    std::cout << "JIT Compilation:     " << compile_duration / 1000.0 << " ms\n";
    std::cout << "Compiled Execution:  " << exec_duration / 1000.0 << " ms\n";
    std::cout << "Total Time:          " << (eager_duration + compile_duration + exec_duration) / 1000.0 << " ms\n";
    
    if (exec_duration > 0) {
        float speedup = static_cast<float>(eager_duration) / exec_duration;
        std::cout << "Speedup (Eager/Compiled): " << speedup << "x\n";
    }

    return 0;
}