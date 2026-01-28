
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ad/ag_all.hpp"
#include "ad/runtime/jit_compiler.hpp" // Include JIT Compiler
#include "optim.hpp"
#include "mlp/activation.h"

using namespace ag;
using namespace ag::nn;
using OwnTensor::Tensor;
using OwnTensor::Shape;
using OwnTensor::Dtype;
using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::TensorOptions;

namespace fs = std::filesystem;

// ============================================================================
// DataLoader (Same as before)
// ============================================================================

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(u_int16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ / 2;
        tokens_ = std::min(total_tokens, max_tokens);

        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    void close() {
        if (data_) {
            ::munmap(data_, file_bytes_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    size_t size_tokens() const { return tokens_; }
    
    void read_block(size_t start, size_t count, std::vector<u_int16_t>& out) const {
        if (start + count > tokens_) throw std::out_of_range("read_block out of range");
        out.resize(count);
        const u_int16_t* p = reinterpret_cast<const u_int16_t*>(data_);
        for (size_t i = 0; i < count; ++i) out[i] = p[start + i];
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};

struct Batch {
    int B = 0, T = 0;
    Tensor input;
    Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T,
                   int rank, int world_size,
                   const std::string& split,
                   const std::string& data_root,
                   bool master_process = true,
                   size_t max_tokens_per_shard = 104457600)
        : B_(B), T_(T),
          rank_(rank), world_(world_size),
          split_(split), root_(data_root),
          max_tokens_(max_tokens_per_shard) {

        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty()) throw std::runtime_error("no .bin shards");
        reset();
    }

    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1;

        if (pos_ + need > shard_.size_tokens()) {
            current_shard_ = (current_shard_ + 1) % shards_.size();
            shard_.open(shards_[current_shard_], max_tokens_);
            pos_ = BT * static_cast<size_t>(rank_);
        }

        std::vector<u_int16_t> buf;
        shard_.read_block(pos_, need, buf);
        pos_ += BT * static_cast<size_t>(world_);

        std::vector<uint16_t> x_vec(BT), y_vec(BT);
        for (size_t i = 0; i < BT; ++i) {
            x_vec[i] = buf[i];
            y_vec[i] = buf[i + 1];
        }

        // Create CPU tensors - JIT will take them as inputs
        Batch b;
        b.B = B_; b.T = T_;
        b.input = Tensor(Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, Dtype::UInt16, Device::CPU);
        std::copy(x_vec.begin(), x_vec.end(), b.input.data<uint16_t>());
        
        b.target = Tensor(Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, Dtype::UInt16, Device::CPU);
        std::copy(y_vec.begin(), y_vec.end(), b.target.data<uint16_t>());

        return b;
    }

private:
    int B_, T_, rank_, world_;
    std::string split_, root_;
    size_t max_tokens_;
    std::vector<std::string> shards_;
    size_t current_shard_ = 0, pos_ = 0;
    UInt16ShardView shard_;
};

// ============================================================================
// Model
// ============================================================================

struct GPTConfig {
    int context_length = 1024;
    int vocab_size = 50304;
    int n_embd = 384; 
    int n_layers = 1; // Testing 1 layer for stability first
    float max_lr = 1e-4f;
    float min_lr = 1e-5f;
    int warmup_steps = 10;
    int max_steps = 100;
};

class TANH : public Module {
public:
    Value operator()(Value x) override { return ag::tanh(x); }
};

class MLP : public Module {
public:
    MLP(GPTConfig config) {
        l_up = new Linear(config.n_embd, 4 * config.n_embd, Device::CUDA); // On GPU
        l_down = new Linear(4 * config.n_embd, config.n_embd, Device::CUDA);
        tanh = new TANH();

        for(auto& p : l_up->parameters()) params_.push_back(p);
        for(auto& p : l_down->parameters()) params_.push_back(p);
    }
    Value operator()(Value x) override {
        // JIT Compiler now supports Op::Linear via automatic decomposition!
        x = (*l_up)(x);
        x = (*tanh)(x);
        x = (*l_down)(x);
        return x;
    }
    Linear *l_up, *l_down;
    TANH* tanh;
};

class GPT_JIT : public Module {
public:
    GPT_JIT(GPTConfig config) : config(config) {
        // Embeddings on GPU
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);
        Shape shape_e(std::vector<int64_t>{(long)config.vocab_size, (long)config.n_embd});
        Shape shape_p(std::vector<int64_t>{(long)config.context_length, (long)config.n_embd});
        
        // Fix: Explicit randn args and set grad manually if needed (or assume default)
        Tensor wte_t = Tensor::randn(shape_e, opts, 1.0f);
        Tensor wpe_t = Tensor::randn(shape_p, opts, 1.0f);
        
        // requires_grad=true is not an arg to randn, but property of Tensor?
        // Let's assume we wrap it in make_tensor which makes it a param (Leaf).
        // If Tensor constructor has requires_grad, randn might not.
        // But parameters_t() returns tensors.
        // We'll trust make_tensor to handle it or wte.val() is the tensor.
        
        wte = make_tensor(wte_t, "wte");
        wpe = make_tensor(wpe_t, "wpe");
        
        // Scale initialization
        wte.val() *= 0.02f;
        wpe.val() *= 0.02f;
        
        params_.push_back(wte);
        params_.push_back(wpe);
        
        mlp = new MLP(config);
        for(auto& p : mlp->parameters()) params_.push_back(p);
    }


    // Implement abstract method from Module
    Value operator()(Value input) override {
        throw std::runtime_error("Use build_jit_graph instead");
    }

    // Wrapper struct to return inputs and outputs clearly
    struct Graph {
        Value input_ids;
        Value target_ids;
        Value logits;
        Value loss;
        std::vector<Value> inputs() { return {input_ids, target_ids}; }
        std::vector<Value> outputs() { return {logits, loss}; }
    };

    Graph build_jit_graph(const Tensor& input_template, const Tensor& target_template) {
        Value input_ids = make_tensor(input_template, "input_ids");
        Value target_ids = make_tensor(target_template, "target_ids");

        // 2. Embeddings via Gather
        // Fix: Scalar creation via full
        Tensor z_t = OwnTensor::Tensor::zeros(Shape(std::vector<int64_t>{1}), ag::options(input_template)); 
        Value zero = make_tensor(z_t, "zero");
        
        Value tok_emb = ag::gather(wte, zero, input_ids);
        
        // Positional IDs
        std::vector<float> pos_data(config.context_length);
        for(int i=0; i<config.context_length; ++i) pos_data[i] = (float)i;
        
        // Fix: Explicit Shape
        Tensor pos_ids_t = Tensor(Shape(std::vector<int64_t>{1, (long)config.context_length}), Dtype::Float32, Device::CUDA);
        std::copy(pos_data.begin(), pos_data.end(), pos_ids_t.to_cpu().data<float>());
        pos_ids_t = pos_ids_t.to(Device::CUDA);
        Value pos_ids_const = make_tensor(pos_ids_t, "pos_ids_const"); 
        
        Value pos_emb = ag::gather(wpe, zero, pos_ids_const); 
        
        Value x = ag::add(tok_emb, pos_emb); 
        
        // 3. MLP
        x = (*mlp)(x);

        // 4. Output Projection
        Value logits = ag::matmul(x, ag::transpose(wte));

        // 5. Loss
        Value loss = ag::sparse_cross_entropy_with_logits(logits, target_ids);
        
        return {input_ids, target_ids, logits, loss};
    }

    GPTConfig config;
    Value wte, wpe;
    MLP* mlp;
};


int main() {
    try {
        std::cout << "===== GPT-2 JIT Training Test (Full Graph) =====\n";
        
        GPTConfig config;
        config.n_layers = 1; 
        config.context_length = 64; 
        config.n_embd = 64;
        
        GPT_JIT model(config);
        
        std::cout << "Model initialized.\n";
        
        int B = 2;
        int T = config.context_length;
        
        // Create Template Tensors for compilation
        // Fix: Explicit std::vector for Shape
        Shape shape_BT(std::vector<int64_t>{(long)B, (long)T});
        TensorOptions opts_f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);
        
        Tensor input_template = Tensor(shape_BT, opts_f32); 
        Tensor target_template = Tensor(shape_BT, opts_f32);

        // Build Graph
        std::cout << "Building Graph...\n";
        auto graph = model.build_jit_graph(input_template, target_template);
        
        // Compile
        std::cout << "Compiling JIT...\n";
        ag::jit::CompileOptions opts;
        opts.include_backward = true; 
        
        auto compiled_fn = ag::jit::compile(graph.loss, graph.inputs(), model.parameters(), opts);
        
        std::cout << "Compilation Success! Running forward/backward...\n";
        
        // Mock Data
        // Fix: Provide explicit standard deviation for randn
        Tensor input_real = Tensor::randn(shape_BT, opts_f32, 1.0f); 
        Tensor target_real = Tensor::randn(shape_BT, opts_f32, 1.0f);
        
        std::vector<Tensor*> runtime_inputs = {&input_real, &target_real};
        
        // Extract parameter tensors
        std::vector<Tensor*> runtime_params;
        auto params = model.parameters();
        runtime_params.reserve(params.size());
        for(auto& p : params) {
            runtime_params.push_back(&p.val());
        }
        
        // Run
        Tensor loss_out;
        if (!compiled_fn.run(runtime_inputs, runtime_params, loss_out)) {
             throw std::runtime_error("JIT Execution Failed");
        }
        
        std::cout << "JIT Run Complete. Loss: " << loss_out.to_cpu().data<float>()[0] << "\n";
        
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
