#include <iostream>
#include <random>
#include <iomanip>
#include "ad/runtime/jit_compiler.hpp"
#include "ad/ag_all.hpp" // Main umbrella header

using namespace ag;

int main() {
    // For pretty printing
    std::cout << std::fixed << std::setprecision(4);

    // ---------- Shapes ----------
    const int B = 8;     // batch size
    const int In = 16;   // input dim
    const int H1 = 64;
    const int H2 = 64;
    const int H3 = 32;
    const int H4 = 32;
    const int Out = 10;  // number of classes

    // ---------- Data ----------
    // Inputs are constants (requires_grad=false, which is the default)
    Value  X  = make_tensor(Tensor::randn<float>(Shape{{B, In}}, TensorOptions()), "X");

    // One-hot labels Y[B,Out]
    Tensor Yt(Shape{{B, Out}}, TensorOptions());
    float* yt_data = Yt.data<float>(); // Get data pointer to fill
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, Out - 1);
    for (int i = 0; i < B; ++i) {
        int k = pick(gen);
        for (int j = 0; j < Out; ++j) {
            yt_data[i * Out + j] = (j == k) ? 1.0f : 0.0f; // Use linear indexing
        }
    }
    Value Y = make_tensor(Yt, "Y_target");

    // ---------- Parameters ----------
    // Parameters are trainable (requires_grad=true)
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn<float>(Shape{{In, H1}}, opts_param) * 0.1f, "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, H1}}, opts_param), "b1");
    auto W2 = make_tensor(Tensor::randn<float>(Shape{{H1, H2}}, opts_param) * 0.1f, "W2");
    auto b2 = make_tensor(Tensor::zeros(Shape{{1, H2}}, opts_param), "b2");
    auto W3 = make_tensor(Tensor::randn<float>(Shape{{H2, H3}}, opts_param) * 0.1f, "W3");
    auto b3 = make_tensor(Tensor::zeros(Shape{{1, H3}}, opts_param), "b3");
    auto W4 = make_tensor(Tensor::randn<float>(Shape{{H3, H4}}, opts_param) * 0.1f, "W4");
    auto b4 = make_tensor(Tensor::zeros(Shape{{1, H4}}, opts_param), "b4");
    auto W5 = make_tensor(Tensor::randn<float>(Shape{{H4, Out}}, opts_param) * 0.1f, "W5");
    auto b5 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b5");
    
    // ---------- Forward: 4 hidden layers + logits ----------
    Value L1 = gelu(matmul(X,  W1) + b1);
    Value L2 = gelu(matmul(L1, W2) + b2);
    Value L3 = gelu(matmul(L2, W3) + b3);
    Value L4 = gelu(matmul(L3, W4) + b4);
    Value logits = matmul(L4, W5) + b5;
    Value loss = mse_loss(logits, Y);


    // ---------- Eager Forward + Backward ----------
    ag::backward(loss);
    float eager_loss = loss.val().to_cpu().data<float>()[0];
    std::cout << "Eager Loss = " << eager_loss << "\n";

    // ---------- JIT Compile ----------
    std::vector<Value> inputs = {X, Y};  // Include Y as input
    std::vector<Value> params;
    params.push_back(W1); params.push_back(b1);
    params.push_back(W2); params.push_back(b2);
    params.push_back(W3); params.push_back(b3);
    params.push_back(W4); params.push_back(b4);
    params.push_back(W5); params.push_back(b5);
    
    std::cout << "\nCompiling graph...\n";
    ag::jit::CompileOptions opts;
    opts.include_backward = true;
    auto compiled = ag::jit::compile(loss, inputs, params, opts);
    std::cout << "Graph compilation successful.\n";

    // ---------- Run Compiled Function ----------
    std::cout << "\nRunning compiled graph...\n";
    
    // Prepare input tensors
    Tensor X_tensor = X.val();
    Tensor Y_tensor = Y.val();
    std::vector<Tensor*> input_ptrs = {&X_tensor, &Y_tensor};
    
    // Prepare param tensors
    Tensor W1_t = W1.val(), b1_t = b1.val();
    Tensor W2_t = W2.val(), b2_t = b2.val();
    Tensor W3_t = W3.val(), b3_t = b3.val();
    Tensor W4_t = W4.val(), b4_t = b4.val();
    Tensor W5_t = W5.val(), b5_t = b5.val();
    std::vector<Tensor*> param_ptrs = {&W1_t, &b1_t, &W2_t, &b2_t, &W3_t, &b3_t, &W4_t, &b4_t, &W5_t, &b5_t};
    
    std::vector<Tensor> jit_outputs;
    if (!compiled.run(input_ptrs, param_ptrs, jit_outputs)) {
        std::cerr << "FAIL: JIT execution failed.\n";
        return 1;
    }
    
    std::cout << "Compiled execution successful.\n";
    std::cout << "JIT returned " << jit_outputs.size() << " outputs.\n";

    // ---------- Verification ----------
    float jit_loss = jit_outputs[0].to_cpu().data<float>()[0];
    
    std::cout << "\n--- Verification ---\n";
    std::cout << "Eager Loss:    " << eager_loss << "\n";
    std::cout << "Compiled Loss: " << jit_loss << "\n";
    
    bool pass = true;
    if (std::abs(eager_loss - jit_loss) < 1e-3f) {
        std::cout << "✅ PASS: Loss matches.\n";
    } else {
        std::cout << "❌ FAIL: Loss mismatch.\n";
        pass = false;
    }
    
    // Verify first param gradient (W1)
    if (jit_outputs.size() > 1) {
        Tensor eager_grad = W1.grad().to_cpu();
        Tensor jit_grad = jit_outputs[1].to_cpu();
        float mad = OwnTensor::reduce_mean(OwnTensor::abs(eager_grad - jit_grad, ag::current_stream())).data<float>()[0];
        std::cout << "W1 Grad MAD: " << mad << "\n";
        if (mad > 1e-3f) pass = false;
    }
    
    if (pass) {
        std::cout << "✅ PASS: All verifications passed.\n";
        return 0;
    } else {
        std::cout << "❌ FAIL: Some verifications failed.\n";
        return 1;
    }
}