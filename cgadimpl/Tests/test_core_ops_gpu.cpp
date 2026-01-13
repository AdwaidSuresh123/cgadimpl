#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <functional>

using namespace ag;
using namespace OwnTensor;

// ==========================================================
// TEST DISPATCHER
// ==========================================================

bool run_test(const std::string& name, std::function<void()> test_func) {
    std::cout << "Testing: " << std::left << std::setw(20) << name << "... \n";
    try {
        test_func();
        std::cout << "[PASS]\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "[FAIL]\n";
        std::cerr << "  Exception: " << e.what() << "\n";
        return false;
    }
}

// ==========================================================
// TEST DEFINITIONS
// ==========================================================

void test_selected_ops() {
    #ifdef WITH_CUDA
        Device dev = Device::CUDA;
    #else
        std::cout << "Skipping GPU tests (WITH_CUDA not defined)\n";
        return;
    #endif
    auto opts = TensorOptions().with_device(dev).with_req_grad(true);

    // // 1. MatMul
    // run_test("MatMul", [&](){
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value b = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value c = matmul(a, b);
    //     backward(sum(c));
        
    //     std::cout << "  MatMul Values:\n";
    //     debug::print_all_values(c);
    //     std::cout << "  MatMul grads:\n";
    //     debug::print_all_grads(c);
    // });

    // // 2. Linear (MatMul + Bias)
    // run_test("Linear", [&](){
    //     Value x = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value w = make_tensor(Tensor::randn(Shape{{4, 4}}, opts)); // (out, in)
    //     Value b = make_tensor(Tensor::randn(Shape{{1, 4}}, opts));
    //     Value y = linear(x, w, b);
    //     backward(sum(y));
        
    //     std::cout << "  Linear grads:\n";
    //     debug::print_all_grads(y);
    //     std::cout << "  Linear values:\n";
    //     debug::print_all_values(y);
    // });

    // // 3. FMAB (Fused Multiply Add Bias: a*b + c)
    // run_test("FMAB", [&](){
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value b = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value c = make_tensor(Tensor::randn(Shape{{1, 4}}, opts));
    //     Value y = fmab(a, b, c);
    //     backward(sum(y));
        
    //     std::cout << "  FMAB grads:\n";
    //     debug::print_all_grads(y);
    //     std::cout << "  FMAB values:\n";
    //     debug::print_all_values(y);
    // });

    // // 4. Tanh
    // run_test("Tanh", [&](){
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value y = tanh(a);
    //     backward(sum(y));
        
    //     std::cout << "  Tanh grads:\n";
    //     debug::print_all_grads(y);
    //     std::cout << "  Tanh values:\n";
    //     debug::print_all_values(y);
    // });

    // run_test("Power", [&](){
    //     Value a = make_tensor(Tensor::rand(Shape{{4, 4}}, opts));
    //     Value b = make_tensor(Tensor::rand(Shape{{4, 4}}, opts));
    //     Value c = pow(a, b);
    //     backward(sum(c));
        
    //     std::cout << "  Power Values:\n";
    //     debug::print_all_values(c);
    //     std::cout << "  Power grads:\n";
    //     debug::print_all_grads(c);
    // });

    // run_test("Sign", [&](){
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value b = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value c = sign(a);
    //     backward(sum(c));
        
    //     std::cout << "  Sign Values:\n";
    //     debug::print_all_values(c);

    //     std::cout << "  Sign grads:\n";
    //     debug::print_all_grads(c);

    // });

// Loss Functions
    
    // run_test("MSE Loss", [&](){
    //     Value y = make_tensor(Tensor::ones(Shape{{4, 4}}, opts));
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value b = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value x = make_tensor(Tensor::randn(Shape{{1, 4}}, opts));
    //     Value c = linear(a, b, x);
    //     Value l = mse_loss(c, y);
    //     backward(l);
        
    //     std::cout << "  MSE Values:\n";
    //     debug::print_all_values(l);

    //     std::cout << "  MSE grads:\n";
    //     debug::print_all_grads(l);

    // });

    //     run_test("MAE Loss", [&](){
    //     Value y = make_tensor(Tensor::ones(Shape{{4, 4}}, opts));
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value b = make_tensor(Tensor::randn(Shape{{4, 4}}, opts));
    //     Value x = make_tensor(Tensor::randn(Shape{{1, 4}}, opts));
    //     Value c = linear(a, b, x);
    //     Value l = mae_loss(c, y);
    //     backward(l);
        
    //     std::cout << "  MAE Values:\n";
    //     debug::print_all_values(l);

    //     std::cout << "  MAE grads:\n";
    //     debug::print_all_grads(l);

    // });

    run_test("Sparse Cross Entropy", [&](){
        Value y = make_tensor(Tensor::ones(Shape{{4, 4}}, opts));
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 4}}, opts));
        Value b = make_tensor(Tensor::randn<float>(Shape{{4, 4}}, opts));
        Value x = make_tensor(Tensor::randn<float>(Shape{{1, 4}}, opts));
        Value c = linear(a, b, x);
        Value l = sparse_cross_entropy_with_logits(c, y);
        backward(l);
        
        std::cout << "  Sparse Cross Entropy Values:\n";
        debug::print_all_values(l);

        std::cout << "  Sparse Cross Entropy grads:\n";
        debug::print_all_grads(l);

    });

}

int main() {
    try {
        #ifndef WITH_CUDA
            std::cout << "Test skipped: Not compiled with CUDA support.\n";
            return 0;
        #endif

        test_selected_ops();

        std::cout << "\n  All tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "\nCaught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}