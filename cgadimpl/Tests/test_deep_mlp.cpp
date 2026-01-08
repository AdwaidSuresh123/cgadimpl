#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include "ad/runtime/jit_compiler.hpp"

using namespace ag;

// ===================================================================
// Test 3-Layer MLP
// ===================================================================

void test_3layer_mlp() {
    std::cout << "\n========================================\n";
    std::cout << "  TEST: 3-Layer MLP\n";
    std::cout << "========================================\n";
    
    const int B = 8;    // batch size
    const int In = 16;  // input dim
    const int H1 = 12;  // hidden1 dim
    const int H2 = 10;  // hidden2 dim
    const int Out = 8;  // output dim
    
    std::cout << "Architecture: " << In << " -> " << H1 << " -> " << H2 << " -> " << Out << "\n";
    
    // Input
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");
    
    // Parameters (6 total: W1, b1, W2, b2, W3, b3)
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn(Shape{{In, H1}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, H1}}, opts_param), "b1");
    auto W2 = make_tensor(Tensor::randn(Shape{{H1, H2}}, opts_param), "W2");
    auto b2 = make_tensor(Tensor::zeros(Shape{{1, H2}}, opts_param), "b2");
    auto W3 = make_tensor(Tensor::randn(Shape{{H2, Out}}, opts_param), "W3");
    auto b3 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b3");
    
    // Forward pass
    // Layer 1: X @ W1 + b1
    Value H1_val = matmul(X, W1) + b1;
    
    // Layer 2: H1 @ W2 + b2
    Value H2_val = matmul(H1_val, W2) + b2;
    
    // Layer 3: H2 @ W3 + b3
    Value Out_val = matmul(H2_val, W3) + b3;
    
    // Loss
    Value loss = sum(Out_val);
    
    std::cout << "\nEager forward pass completed.\n";
    float eager_val = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << eager_val << "\n";
    
    // Compile
    std::cout << "\nCompiling 3-layer MLP...\n";
    std::vector<Value> inputs = {X};
    std::vector<Value> params = {W1, b1, W2, b2, W3, b3};
    std::cout << "Total arguments: 1 result + 1 input + " << params.size() << " params = " 
              << (1 + 1 + params.size()) << " arguments\n";
    
    auto comp = ag::jit::compile(loss, inputs, params);
    std::cout << "Compilation successful.\n";
    
    // Execute
    std::cout << "\nRunning compiled 3-layer MLP...\n";
    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value,
                                      &W2.node->value, &b2.node->value,
                                      &W3.node->value, &b3.node->value};
    
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
        std::cout << "✅ PASS: 3-Layer MLP test passed!\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }
}

// ===================================================================
// Test 4-Layer MLP
// ===================================================================

void test_4layer_mlp() {
    std::cout << "\n========================================\n";
    std::cout << "  TEST: 4-Layer MLP\n";
    std::cout << "========================================\n";
    
    const int B = 8;    // batch size
    const int In = 16;  // input dim
    const int H1 = 14;  // hidden1 dim
    const int H2 = 12;  // hidden2 dim
    const int H3 = 10;  // hidden3 dim
    const int Out = 8;  // output dim
    
    std::cout << "Architecture: " << In << " -> " << H1 << " -> " << H2 
              << " -> " << H3 << " -> " << Out << "\n";
    
    // Input
    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");
    
    // Parameters (8 total: W1, b1, W2, b2, W3, b3, W4, b4)
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn(Shape{{In, H1}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, H1}}, opts_param), "b1");
    auto W2 = make_tensor(Tensor::randn(Shape{{H1, H2}}, opts_param), "W2");
    auto b2 = make_tensor(Tensor::zeros(Shape{{1, H2}}, opts_param), "b2");
    auto W3 = make_tensor(Tensor::randn(Shape{{H2, H3}}, opts_param), "W3");
    auto b3 = make_tensor(Tensor::zeros(Shape{{1, H3}}, opts_param), "b3");
    auto W4 = make_tensor(Tensor::randn(Shape{{H3, Out}}, opts_param), "W4");
    auto b4 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b4");
    
    // Forward pass
    // Layer 1: X @ W1 + b1
    Value H1_val = matmul(X, W1) + b1;
    
    // Layer 2: H1 @ W2 + b2
    Value H2_val = matmul(H1_val, W2) + b2;
    
    // Layer 3: H2 @ W3 + b3
    Value H3_val = matmul(H2_val, W3) + b3;
    
    // Layer 4: H3 @ W4 + b4
    Value Out_val = matmul(H3_val, W4) + b4;
    
    // Loss
    Value loss = sum(Out_val);
    
    std::cout << "\nEager forward pass completed.\n";
    float eager_val = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss: " << eager_val << "\n";
    
    // Compile
    std::cout << "\nCompiling 4-layer MLP...\n";
    std::vector<Value> inputs = {X};
    std::vector<Value> params = {W1, b1, W2, b2, W3, b3, W4, b4};
    std::cout << "Total arguments: 1 result + 1 input + " << params.size() << " params = " 
              << (1 + 1 + params.size()) << " arguments\n";
    
    auto comp = ag::jit::compile(loss, inputs, params);
    std::cout << "Compilation successful.\n";
    
    // Execute
    std::cout << "\nRunning compiled 4-layer MLP...\n";
    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W1.node->value, &b1.node->value,
                                      &W2.node->value, &b2.node->value,
                                      &W3.node->value, &b3.node->value,
                                      &W4.node->value, &b4.node->value};
    
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
    
    if (error < 1e-3f) {  // Slightly relaxed tolerance for deeper networks
        std::cout << "✅ PASS: 4-Layer MLP test passed!\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }
}

// ===================================================================
// Main
// ===================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  DEEP MLP TESTS (libffi validation)\n";
    std::cout << "========================================\n";
    
    test_3layer_mlp();
    test_4layer_mlp();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    
    return 0;
}
