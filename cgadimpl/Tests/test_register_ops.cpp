#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

using namespace ag;
using namespace std;
using namespace OwnTensor;
using namespace mlp_forward;

// Helper to print pass/fail
void assert_true(bool condition, const string& message) {
    if (condition) {
        cout << "[PASS] " << message << endl;
    } else {
        cout << "[FAIL] " << message << endl;
        exit(1);
    }
}

void assert_close(float actual, float expected, const string& message, float tol = 1e-4) {
    if (std::abs(actual - expected) < tol) {
        cout << "[PASS] " << message << " (Value: " << actual << ")" << endl;
    } else {
        cout << "[FAIL] " << message << " (Expected: " << expected << ", Actual: " << actual << ")" << endl;
        exit(1);
    }
}

// Helper to check if gradient is computed (non-zero/non-empty)
void check_grad(Value& v, const string& name) {
    if (v.node && v.node->requires_grad()) {
        // Simple check: gradient should be allocated and have same shape
        bool has_grad = v.grad().numel() > 0;
        assert_true(has_grad, "Gradient computed for " + name);
    }
}

void test_activations() {
    cout << "\n--- Testing Activations ---\n";
    
    Tensor data = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
    Value x = make_tensor(data, "x");

    // 1. GELU
    {
        Value y = gelu(x);
        backward(sum(y)); // simple scalar backward
        check_grad(x, "GELU input");
        zero_grad(x);
        cout << "GELU test complete.\n";
    }

    // 2. ReLU
    {
        Value y = relu(x);
        backward(sum(y));
        check_grad(x, "ReLU input");
        zero_grad(x);
        cout << "ReLU test complete.\n";
    }

    // 3. Sigmoid
    {
        Value y = sigmoid(x);
        backward(sum(y));
        check_grad(x, "Sigmoid input");
        zero_grad(x);
        cout << "Sigmoid test complete.\n";
    }

    // 4. Softmax
    {
        Value y = softmax_row(x); // default axis -1
        backward(sum(y));
        check_grad(x, "Softmax input");
        zero_grad(x);
        cout << "Softmax test complete.\n";
    }

    // 5. Tanh
    {
        Value y = tanh(x);
        backward(sum(y));
        check_grad(x, "Tanh input");
        zero_grad(x);
        cout << "Tanh test complete.\n";
    }
}

void test_layers() {
    cout << "\n--- Testing Layers ---\n";

    // 1. Linear
    {
        Tensor x_data = Tensor::randn<float>(Shape{{2, 4}}, TensorOptions().with_req_grad(true));
        Value x = make_tensor(x_data, "x");
        nn::Linear linear(4, 2);
        
        Value y = linear(x);
        backward(sum(y));
        
        check_grad(x, "Linear input");
        // Check weights and bias grads
        for(auto& p : linear.parameters()) {
            check_grad(p, "Linear parameter");
        }
        cout << "Linear test complete.\n";
    }

    // Dropout
    {
        Tensor x_data = Tensor::randn<float>(Shape{{5, 5}}, TensorOptions().with_req_grad(true));
        Value x = make_tensor(x_data, "x");
        Tensor mask_data = Tensor::ones(x_data.shape(), TensorOptions());
        Value mask = make_tensor(mask_data, "mask");
        
        Value y = dropout(x, mask);
        backward(sum(y));
        check_grad(x, "Dropout input");
        cout << "Dropout test complete.\n";
    }

    // Flatten
    {
        Tensor x_data = Tensor::randn<float>(Shape{{2, 3, 4}}, TensorOptions().with_req_grad(true));
        Value x = make_tensor(x_data, "x");
        
        Value y = flatten(x);
        assert_true(y.shape().size() == 2, "Flatten output rank is 2");
        assert_true(y.shape()[0] == 2 && y.shape()[1] == 12, "Flatten output shape correct");
        
        backward(sum(y));
        check_grad(x, "Flatten input");
        cout << "Flatten test complete.\n";
    }
}

void test_losses() {
    cout << "\n--- Testing Losses ---\n";
    
    Tensor pred_data = Tensor::rand<float>(Shape{{2, 2}}, TensorOptions().with_req_grad(true)); // 0-1 for BCE
    Tensor target_data = Tensor::rand<float>(Shape{{2, 2}}, TensorOptions());
    
    Value pred = make_tensor(pred_data, "pred");
    Value target = make_tensor(target_data, "target");

    // 1. Binary Cross Entropy
    {
        Value loss = binary_cross_entropy(pred, target);
        std::cout << "Binary Cross Entropy Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(pred, "BCE input");
        zero_grad(pred);
        cout << "BCE test complete.\n";
    }

    // 2. Categorical Cross Entropy
    {
        // Preds should be probabilities summing to 1 for CCE usually, but for test we just check grad flow
        // Using softmax to ensure valid inputs
        Value sm_pred = softmax_row(pred); 
        Value loss = categorical_cross_entropy(sm_pred, target);
        std::cout << "Categorical Cross Entropy Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(pred, "CCE input"); // Gradient should flow back to pred through softmax
        zero_grad(pred);
        cout << "CCE test complete.\n";
    }

    // 3. MAE Loss
    {
        Value loss = mae_loss(pred, target);
        std::cout << "MAE Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(pred, "MAE input");
        zero_grad(pred);
        cout << "MAE test complete.\n";
    }

    // 4. MSE Loss
    {
        Value loss = mse_loss(pred, target);
        std::cout << "MSE Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(pred, "MSE input");
        zero_grad(pred);
        cout << "MSE test complete.\n";
    }

    // 5. Sparse CE With Logits
    {
        cout << "Starting Sparse CE test...\n";
        Tensor logits_data = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
        Tensor labels_data = Tensor::zeros(Shape{{2}}, TensorOptions().with_dtype(Dtype::Int32));
        labels_data.data<int32_t>()[0] = 1;
        labels_data.data<int32_t>()[1] = 2;

        Value logits = make_tensor(logits_data, "logits");
        Value labels = make_tensor(labels_data, "labels");

        Value loss = sparse_cross_entropy_with_logits(logits, labels);
        std::cout << "Sparse CE Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(logits, "Sparse CE input");
        zero_grad(logits);
        cout << "Sparse CE test complete.\n";
    }

    // 6. KL Divergence
    {
        cout << "Starting KL Divergence test...\n";
        Tensor p_data = Tensor::rand<float>(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
        Tensor q_data = Tensor::rand<float>(Shape{{2, 3}}, TensorOptions());
        
        Value p = make_tensor(p_data, "p");
        Value q = make_tensor(q_data, "q");
        
        Value loss = kldivergence(p, q);
        std::cout << "KL Divergence Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(p, "KL Div input");
        zero_grad(p);
        cout << "KL Divergence test complete.\n";
    }

    // 7. Cross Entropy with Logits
    {
        cout << "Starting CE with Logits test...\n";
        Tensor logits_data = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
        Tensor target_data = Tensor::zeros(Shape{{2, 3}}, TensorOptions());
        target_data.data<float>()[1] = 1.0f; // one-hot
        target_data.data<float>()[5] = 1.0f;
        
        Value logits = make_tensor(logits_data, "logits");
        Value target = make_tensor(target_data, "target");
        
        Value loss = cross_entropy_with_logits(logits, target);
        std::cout << "CE with Logits Loss: ";
        loss.val().display();
        backward(loss);
        check_grad(logits, "CE with Logits input");
        zero_grad(logits);
        cout << "CE with Logits test complete.\n";
    }
}

void test_losses_correctness() {
    cout << "\n--- Testing Losses Correctness ---\n";

    // 1. MSE Loss
    {
        Tensor pred_data = Tensor::ones(Shape{{2}}, TensorOptions());
        pred_data.data<float>()[0] = 2.0f;
        pred_data.data<float>()[1] = 4.0f;
        
        Tensor target_data = Tensor::ones(Shape{{2}}, TensorOptions());
        target_data.data<float>()[0] = 1.0f;
        target_data.data<float>()[1] = 3.0f;

        Value pred = make_tensor(pred_data, "pred");
        Value target = make_tensor(target_data, "target");
        
        Value loss = mse_loss(pred, target);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 1.0f, "MSE Loss correctness");
    }

    // 2. MAE Loss
    {
        Tensor pred_data = Tensor::ones(Shape{{2}}, TensorOptions());
        pred_data.data<float>()[0] = 2.0f;
        pred_data.data<float>()[1] = 4.0f;
        
        Tensor target_data = Tensor::ones(Shape{{2}}, TensorOptions());
        target_data.data<float>()[0] = 1.0f;
        target_data.data<float>()[1] = 5.0f;

        Value pred = make_tensor(pred_data, "pred");
        Value target = make_tensor(target_data, "target");
        
        Value loss = mae_loss(pred, target);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 1.0f, "MAE Loss correctness");
    }

    // 3. BCE Loss
    {
        Tensor pred_data = Tensor::ones(Shape{{2}}, TensorOptions());
        pred_data.data<float>()[0] = 0.5f;
        pred_data.data<float>()[1] = 0.5f;
        
        Tensor target_data = Tensor::ones(Shape{{2}}, TensorOptions());
        target_data.data<float>()[0] = 1.0f;
        target_data.data<float>()[1] = 0.0f;

        Value pred = make_tensor(pred_data, "pred");
        Value target = make_tensor(target_data, "target");
        
        Value loss = binary_cross_entropy(pred, target);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 0.693147f, "BCE Loss correctness");
    }

    // 4. CCE Loss
    {
        Tensor pred_data = Tensor::ones(Shape{{1, 2}}, TensorOptions());
        pred_data.data<float>()[0] = 0.5f;
        pred_data.data<float>()[1] = 0.5f;
        
        Tensor target_data = Tensor::ones(Shape{{1, 2}}, TensorOptions());
        target_data.data<float>()[0] = 1.0f;
        target_data.data<float>()[1] = 0.0f;

        Value pred = make_tensor(pred_data, "pred");
        Value target = make_tensor(target_data, "target");
        
        Value loss = categorical_cross_entropy(pred, target);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 0.693147f, "CCE Loss correctness");
    }

    // 5. Sparse CE Loss
    {
        // Logits: [[0, 1000, 0]] -> Softmax: [[0, 1, 0]]
        // Target: [1] -> Loss: -log(1) = 0
        Tensor logits_data = Tensor::zeros(Shape{{1, 3}}, TensorOptions());
        logits_data.data<float>()[1] = 1000.0f;
        
        Tensor target_data = Tensor::zeros(Shape{{1}}, TensorOptions().with_dtype(Dtype::Int32));
        target_data.data<int32_t>()[0] = 1;

        Value logits = make_tensor(logits_data, "logits");
        Value target = make_tensor(target_data, "target");
        
        Value loss = sparse_cross_entropy_with_logits(logits, target);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 0.0f, "Sparse CE Loss correctness");
    }

    // 6. KL Divergence correctness
    {
        Tensor p_data = Tensor::ones(Shape{{1, 2}}, TensorOptions());
        p_data.data<float>()[0] = 0.5f;
        p_data.data<float>()[1] = 0.5f;
        
        Tensor q_data = Tensor::ones(Shape{{1, 2}}, TensorOptions());
        q_data.data<float>()[0] = 0.5f;
        q_data.data<float>()[1] = 0.5f;

        Value p = make_tensor(p_data, "p");
        Value q = make_tensor(q_data, "q");
        
        Value loss = kldivergence(p, q);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 0.0f, "KL Divergence correctness");
    }

    // 7. CE with Logits correctness
    {
        // Logits: [[0, 1000]] -> Softmax: [[0, 1]]
        // Target: [[0, 1]] -> Loss: -log(1) = 0
        Tensor logits_data = Tensor::zeros(Shape{{1, 2}}, TensorOptions());
        logits_data.data<float>()[1] = 1000.0f;
        
        Tensor target_data = Tensor::zeros(Shape{{1, 2}}, TensorOptions());
        target_data.data<float>()[1] = 1.0f;

        Value logits = make_tensor(logits_data, "logits");
        Value target = make_tensor(target_data, "target");
        
        Value loss = cross_entropy_with_logits(logits, target);
        float loss_val = loss.val().data<float>()[0];
        assert_close(loss_val, 0.0f, "CE with Logits correctness");
    }
}

int main() {
    try {
        //test_activations();
        //test_layers();
        test_losses();
        test_losses_correctness();
        cout << "\nAll registered operations tests PASSED!\n";
    } catch (const std::exception& e) {
        cout << "\n[ERROR] Test failed with exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}