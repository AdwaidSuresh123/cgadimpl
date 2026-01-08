#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "ad/runtime/jit_compiler.hpp"

using namespace ag;

// ===================================================================
// Test Harness
// ===================================================================

struct TestResult {
    std::string name;
    bool passed;
    float eager_result;
    float compiled_result;
    float error;
};

std::vector<TestResult> test_results;

bool verify_results(const std::string& test_name, float eager_val, float compiled_val, float tolerance = 1e-4f) {
    float error = std::abs(eager_val - compiled_val);
    bool passed = error < tolerance;
    
    TestResult result;
    result.name = test_name;
    result.passed = passed;
    result.eager_result = eager_val;
    result.compiled_result = compiled_val;
    result.error = error;
    test_results.push_back(result);
    
    std::cout << "\n--- " << test_name << " ---\n";
    std::cout << "Eager:    " << eager_val << "\n";
    std::cout << "Compiled: " << compiled_val << "\n";
    std::cout << "Error:    " << error << "\n";
    std::cout << (passed ? "✅ PASS" : "❌ FAIL") << "\n";
    
    return passed;
}

void print_summary() {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "         TEST SUMMARY\n";
    std::cout << "========================================\n";
    
    int passed = 0;
    int total = test_results.size();
    
    for (const auto& result : test_results) {
        std::cout << (result.passed ? "✅" : "❌") << " " << result.name;
        if (!result.passed) {
            std::cout << " (error: " << result.error << ")";
        }
        std::cout << "\n";
        if (result.passed) passed++;
    }
    
    std::cout << "========================================\n";
    std::cout << "Total: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n";
}

// ===================================================================
// Test 1: Simple Linear Layer (Baseline)
// ===================================================================

bool test_simple_linear() {
    std::cout << "\n\n=== TEST 1: Simple Linear Layer ===\n";
    
    const int B = 8, In = 16, Out = 10;
    
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");
    
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W = make_tensor(Tensor::randn(Shape{{In, Out}}, opts_param), "W");
    auto b = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b");
    
    // Forward: Z = X @ W + b, loss = sum(Z)
    Value Z = matmul(X, W) + b;
    Value loss = sum(Z);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {X}, {W, b});
    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W.node->value, &b.node->value};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 1: Simple Linear", eager_val, compiled_val);
}

// ===================================================================
// Test 2: Element-wise Operations
// ===================================================================

bool test_elementwise_ops() {
    std::cout << "\n\n=== TEST 2: Element-wise Operations ===\n";
    
    const int M = 4, N = 6;
    
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{M, N}}, opts);
    Tensor Bt = Tensor::randn(Shape{{M, N}}, opts);
    Tensor Ct = Tensor::randn(Shape{{M, N}}, opts);
    
    Value A = make_tensor(At, "A");
    Value B = make_tensor(Bt, "B");
    Value C = make_tensor(Ct, "C");
    
    // Forward: D = (A + B) * C - A
    Value D = (A + B) * C - A;
    Value loss = sum(D);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {A, B, C}, {});
    std::vector<Tensor*> in_ptrs = {&A.node->value, &B.node->value, &C.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 2: Element-wise Ops", eager_val, compiled_val);
}

// ===================================================================
// Test 3: Multi-layer Network (2-layer MLP)
// ===================================================================

bool test_multilayer_network() {
    std::cout << "\n\n=== TEST 3: Multi-layer Network ===\n";
    
    const int B = 8, In = 16, H = 12, Out = 10;
    
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");
    
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn(Shape{{In, H}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, H}}, opts_param), "b1");
    auto W2 = make_tensor(Tensor::randn(Shape{{H, Out}}, opts_param), "W2");
    auto b2 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b2");
    
    // Forward: 2-layer MLP
    // H1 = X @ W1 + b1
    // Out = H1 @ W2 + b2
    // loss = sum(Out)
    Value H1 = matmul(X, W1) + b1;
    Value Out_val = matmul(H1, W2) + b2;
    Value loss = sum(Out_val);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {X}, {W1, b1, W2, b2});
    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value, 
                                      &W2.node->value, &b2.node->value};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 3: Multi-layer Network", eager_val, compiled_val);
}

// ===================================================================
// Test 4: Reduction Operations
// ===================================================================

bool test_reduction_ops() {
    std::cout << "\n\n=== TEST 4: Reduction Operations ===\n";
    
    const int M = 8, N = 10;
    
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{M, N}}, opts);
    Value A = make_tensor(At, "A");
    
    // Test sum reduction
    Value loss = sum(A);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {A}, {});
    std::vector<Tensor*> in_ptrs = {&A.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 4: Reduction (sum)", eager_val, compiled_val);
}

// ===================================================================
// Test 5: Mean Reduction
// ===================================================================

bool test_mean_reduction() {
    std::cout << "\n\n=== TEST 5: Mean Reduction ===\n";
    
    const int M = 6, N = 8;
    
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{M, N}}, opts);
    Value A = make_tensor(At, "A");
    
    // Test mean reduction
    Value loss = mean_all(A);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {A}, {});
    std::vector<Tensor*> in_ptrs = {&A.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 5: Mean Reduction", eager_val, compiled_val);
}

// ===================================================================
// Test 6: Matrix Multiplication Chain
// ===================================================================

bool test_matmul_chain() {
    std::cout << "\n\n=== TEST 6: Matrix Multiplication Chain ===\n";
    
    const int M = 8, K1 = 10, K2 = 12, N = 6;
    
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{M, K1}}, opts);
    Tensor Bt = Tensor::randn(Shape{{K1, K2}}, opts);
    Tensor Ct = Tensor::randn(Shape{{K2, N}}, opts);
    
    Value A = make_tensor(At, "A");
    Value B = make_tensor(Bt, "B");
    Value C = make_tensor(Ct, "C");
    
    // Forward: D = (A @ B) @ C
    Value AB = matmul(A, B);
    Value D = matmul(AB, C);
    Value loss = sum(D);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {A, B, C}, {});
    std::vector<Tensor*> in_ptrs = {&A.node->value, &B.node->value, &C.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 6: MatMul Chain", eager_val, compiled_val);
}

// ===================================================================
// Test 7: Different Batch Sizes
// ===================================================================

bool test_different_batch_sizes() {
    std::cout << "\n\n=== TEST 7: Different Batch Sizes ===\n";
    
    const int B = 16, In = 8, Out = 4;  // Different batch size
    
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");
    
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W = make_tensor(Tensor::randn(Shape{{In, Out}}, opts_param), "W");
    auto b = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b");
    
    // Forward: Z = X @ W + b, loss = sum(Z)
    Value Z = matmul(X, W) + b;
    Value loss = sum(Z);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {X}, {W, b});
    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W.node->value, &b.node->value};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 7: Batch Size 16", eager_val, compiled_val);
}

// ===================================================================
// Test 8: Complex Expression
// ===================================================================

bool test_complex_expression() {
    std::cout << "\n\n=== TEST 8: Complex Expression ===\n";
    
    const int M = 6, N = 8;
    
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{M, N}}, opts);
    Tensor Bt = Tensor::randn(Shape{{M, N}}, opts);
    
    Value A = make_tensor(At, "A");
    Value B = make_tensor(Bt, "B");
    
    // Complex expression: ((A + B) * A - B) + (A * B)
    Value expr1 = (A + B) * A - B;
    Value expr2 = A * B;
    Value result = expr1 + expr2;
    Value loss = sum(result);
    
    float eager_val = loss.val().to_cpu().data<float>()[0];
    
    // Compile and run
    auto comp = ag::jit::compile(loss, {A, B}, {});
    std::vector<Tensor*> in_ptrs = {&A.node->value, &B.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "Compilation/execution failed!\n";
        return false;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    return verify_results("Test 8: Complex Expression", eager_val, compiled_val);
}

// ===================================================================
// Main Test Runner
// ===================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  COMPREHENSIVE ABI TEST SUITE\n";
    std::cout << "========================================\n";
    
    // Run all tests
    test_simple_linear();
    test_elementwise_ops();
    test_multilayer_network();
    test_reduction_ops();
    test_mean_reduction();
    test_matmul_chain();
    test_different_batch_sizes();
    test_complex_expression();
    
    // Print summary
    print_summary();
    
    // Return 0 if all tests passed, 1 otherwise
    int failed = 0;
    for (const auto& result : test_results) {
        if (!result.passed) failed++;
    }
    
    return failed > 0 ? 1 : 0;
}
