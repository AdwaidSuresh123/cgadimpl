#include "ad/ag_all.hpp" // TEST_NN.CPP
#include <iostream>
#include <vector>
#include <iomanip>


int main() {
    using namespace ag;
    using namespace OwnTensor;
    cudaSetDevice(0);
    std::cout << "========================================\n";
    std::cout << "--- Starting End-to-End MLP Training ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Define Hyperparameters ---
    const int batch_size = 16;
    const int input_features = 128;
    const int hidden_features = 64;
    const int output_features = 10;
    const float learning_rate = 0.01f;
    const int epochs = 15;

    // 2. --- Create the Model ---
    ag::nn::Sequential model({
        new ag::nn::Linear(input_features, hidden_features),
        new ag::nn::ReLU(),
        new ag::nn::Linear(hidden_features, output_features)
        // new ag::nn::Linear(hidden_features, output_features)
    });
    std::cout << "Model created with " << model.parameters().size() << " parameter tensors.\n\n";
    model.to(Device::CUDA);
    // 3. --- Generate Random Data ---
    // Tensor x_tensor = 
    Tensor y_tensor = Tensor::randn<float>(Shape{{batch_size, output_features}}, TensorOptions().with_req_grad(true).with_device(Device::CUDA));
    Value X = make_tensor(Tensor::randn<float>(Shape{{batch_size, input_features}}, TensorOptions().with_req_grad(true).with_device(Device::CUDA)), "X_data");
    Value Y = make_tensor(y_tensor, "Y_target");

    // 4. --- The Training Loop ---
    for (int epoch = 0; epoch < 2; ++epoch) {
        Value predictions = model(X);
        Value loss = mse_loss(predictions, Y);
        // loss.val().to_cpu().display();

        float loss_value = loss.val().to_cpu().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch 
                  << ", Loss: " << std::fixed << std::setprecision(4) << loss_value << std::endl;

        model.zero_grad();
        backward(loss);
        model.parameters()[0].grad().to_cpu().display();

        // This loop will now work because Module::parameters() is non-const
        for (Value& param : model.parameters()) {
            if (param.node && param.node->requires_grad()) {
                // This += will now work because param.val() is non-const
                param.val() += (param.grad() * -learning_rate);
            }
        }
    }

    // 5. --- Clean up dynamically allocated modules ---
    // This will now work because get_layers() exists
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n  Training finished successfully.\n";
    return 0;
}