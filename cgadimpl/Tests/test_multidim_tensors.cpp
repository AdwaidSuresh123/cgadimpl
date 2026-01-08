#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include "ad/runtime/jit_compiler.hpp"

using namespace ag;

// ===================================================================
// Test 3D Tensor Reduction
// ===================================================================

void test_3d_reduction() {
    std::cout << "\n========================================\n";
    std::cout << "  TEST: 3D Tensor Reduction\n";
    std::cout << "========================================\n";
    
    const int Batch = 4;
    const int Height = 8;
    const int Width = 6;
    
    std::cout << "Tensor shape: [" << Batch << ", " << Height << ", " << Width << "] (3D)\n";
    
    // Create 3D tensor
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{Batch, Height, Width}}, opts);
    Value A = make_tensor(At, "A");
    
    // Simple reduction
    Value loss = sum(A);
    
    std::cout << "\nEager forward pass completed.\n";
    float eager_val = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << eager_val << "\n";
    
    // Compile
    std::cout << "\nCompiling 3D tensor reduction...\n";
    auto comp = ag::jit::compile(loss, {A}, {});
    std::cout << "Compilation successful.\n";
    
    // Execute
    std::cout << "\nRunning compiled 3D tensor reduction...\n";
    std::vector<Tensor*> in_ptrs = {&A.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "❌ FAIL: Compiled execution failed.\n";
        return;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    std::cout << "Compiled Loss: " << compiled_val << "\n";
    
    // Verify
    std::cout << "\n--- Verification ---\n";
    std::cout << "Eager Result:    " << eager_val << "\n";
    std::cout << "Compiled Result: " << compiled_val << "\n";
    float error = std::abs(eager_val - compiled_val);
    std::cout << "Error:           " << error << "\n";
    
    if (error < 1e-4f) {
        std::cout << "✅ PASS: 3D tensor reduction test passed!\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }
}

// ===================================================================
// Test 4D Tensor Reduction
// ===================================================================

void test_4d_reduction() {
    std::cout << "\n========================================\n";
    std::cout << "  TEST: 4D Tensor Reduction\n";
    std::cout << "========================================\n";
    
    const int Batch = 2;
    const int Channels = 3;
    const int Height = 4;
    const int Width = 5;
    
    std::cout << "Tensor shape: [" << Batch << ", " << Channels << ", " 
              << Height << ", " << Width << "] (4D)\n";
    
    // Create 4D tensor
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{Batch, Channels, Height, Width}}, opts);
    Value A = make_tensor(At, "A");
    
    // Simple reduction
    Value loss = sum(A);
    
    std::cout << "\nEager forward pass completed.\n";
    float eager_val = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << eager_val << "\n";
    
    // Compile
    std::cout << "\nCompiling 4D tensor reduction...\n";
    auto comp = ag::jit::compile(loss, {A}, {});
    std::cout << "Compilation successful.\n";
    
    // Execute
    std::cout << "\nRunning compiled 4D tensor reduction...\n";
    std::vector<Tensor*> in_ptrs = {&A.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "❌ FAIL: Compiled execution failed.\n";
        return;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    std::cout << "Compiled Loss: " << compiled_val << "\n";
    
    // Verify
    std::cout << "\n--- Verification ---\n";
    std::cout << "Eager Result:    " << eager_val << "\n";
    std::cout << "Compiled Result: " << compiled_val << "\n";
    float error = std::abs(eager_val - compiled_val);
    std::cout << "Error:           " << error << "\n";
    
    if (error < 1e-4f) {
        std::cout << "✅ PASS: 4D tensor reduction test passed!\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }
}

// ===================================================================
// Test 5D Tensor Reduction
// ===================================================================

void test_5d_reduction() {
    std::cout << "\n========================================\n";
    std::cout << "  TEST: 5D Tensor Reduction\n";
    std::cout << "========================================\n";
    
    const int D1 = 2;
    const int D2 = 3;
    const int D3 = 4;
    const int D4 = 3;
    const int D5 = 2;
    
    std::cout << "Tensor shape: [" << D1 << ", " << D2 << ", " << D3 
              << ", " << D4 << ", " << D5 << "] (5D)\n";
    
    // Create 5D tensor
    auto opts = TensorOptions();
    Tensor At = Tensor::randn(Shape{{D1, D2, D3, D4, D5}}, opts);
    Value A = make_tensor(At, "A");
    
    // Simple reduction
    Value loss = sum(A);
    
    std::cout << "\nEager forward pass completed.\n";
    float eager_val = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << eager_val << "\n";
    
    // Compile
    std::cout << "\nCompiling 5D tensor reduction...\n";
    auto comp = ag::jit::compile(loss, {A}, {});
    std::cout << "Compilation successful.\n";
    
    // Execute
    std::cout << "\nRunning compiled 5D tensor reduction...\n";
    std::vector<Tensor*> in_ptrs = {&A.node->value};
    std::vector<Tensor*> par_ptrs = {};
    
    Tensor compiled_out;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_out);
    
    if (!ok) {
        std::cerr << "❌ FAIL: Compiled execution failed.\n";
        return;
    }
    
    float compiled_val = compiled_out.to_cpu().data<float>()[0];
    std::cout << "Compiled Loss: " << compiled_val << "\n";
    
    // Verify
    std::cout << "\n--- Verification ---\n";
    std::cout << "Eager Result:    " << eager_val << "\n";
    std::cout << "Compiled Result: " << compiled_val << "\n";
    float error = std::abs(eager_val - compiled_val);
    std::cout << "Error:           " << error << "\n";
    
    if (error < 1e-4f) {
        std::cout << "✅ PASS: 5D tensor reduction test passed!\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }
}

// ===================================================================
// Main
// ===================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  HIGHER-DIMENSIONAL TENSOR TESTS\n";
    std::cout << "  (Testing ABI Adapter with 3D, 4D, 5D)\n";
    std::cout << "========================================\n";
    
    test_3d_reduction();
    test_4d_reduction();
    test_5d_reduction();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    
    return 0;
}
