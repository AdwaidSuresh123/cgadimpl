// GPT-2 C++ Training Script - No PyTorch Dependencies
// Based on trainpy.py architecture with cgadimpl autodiff library
// WORKING VERSION - Using random initialized embeddings without custom backward hooks

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
// DataLoader Implementation
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

    UInt16ShardView(const UInt16ShardView&) = delete;
    UInt16ShardView& operator=(const UInt16ShardView&) = delete;

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
        file_bytes_ = 0;
        tokens_ = 0;
        path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

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
    std::vector<u_int16_t> x;
    std::vector<u_int16_t> y;
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
          master_(master_process),
          max_tokens_(max_tokens_per_shard) {

        if (!(split_ == "train" || split_ == "val"))
            throw std::runtime_error("split must be 'train' or 'val'");
        if (B_ <= 0 || T_ <= 0)
            throw std::runtime_error("B and T must be > 0");
        if (world_ <= 0 || rank_ < 0 || rank_ >= world_)
            throw std::runtime_error("invalid rank/world_size");

        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty())
            throw std::runtime_error("no .bin shards found for split " + split_);

        if (master_) {
            std::cout << "found " << shards_.size() << " shards for split " << split_ << "\n";
        }

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

        if (shard_.size_tokens() < need) {
            throw std::runtime_error("shard too small for one batch: " + shard_.path());
        }

        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        std::vector<u_int16_t> buf;
        shard_.read_block(pos_, need, buf);

        Batch b;
        b.B = B_; b.T = T_;
        b.x.resize(BT);
        b.y.resize(BT);

        for (size_t i = 0; i < BT; ++i) {
            b.x[i] = buf[i];
            b.y[i] = buf[i + 1];
        }

        b.input = Tensor(Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, Dtype::UInt16, Device::CPU);
        std::copy(b.x.begin(), b.x.end(), b.input.data<uint16_t>());
        
        b.target = Tensor(Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, Dtype::UInt16, Device::CPU);
        std::copy(b.y.begin(), b.y.end(), b.target.data<uint16_t>());

        pos_ += BT * static_cast<size_t>(world_);

        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }

        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        shard_.open(shards_[current_shard_], max_tokens_);
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        pos_ = BT * static_cast<size_t>(rank_);
    }

    int B_, T_;
    int rank_, world_;
    std::string split_, root_;
    bool master_;
    size_t max_tokens_;

    std::vector<std::string> shards_;
    size_t current_shard_ = 0;
    size_t pos_ = 0;

    UInt16ShardView shard_;
};

// ============================================================================
// Model Configuration
// ============================================================================

struct GPTConfig {
    int context_length = 1024;  // Reduced for testing
    int vocab_size = 50304;
    int n_embd = 384;          // Reduced for faster testing
    int n_layers = 6;

    float max_lr = 1e-4f;      // Stable LR for training with embeddings
    float min_lr = 1e-5f;
    int warmup_steps = 637;
    int max_steps = 6376;
};

// ============================================================================
// Module Implementations
// ============================================================================

class TANH : public Module {
public:
    Value operator()(Value x) override {
        return ag::tanh(x);
    }
};

class MLP : public Module {
public:
    MLP(GPTConfig config) {
        l_up = new Linear(config.n_embd, 4 * config.n_embd, Device::CPU);
        l_down = new Linear(4 * config.n_embd, config.n_embd, Device::CPU);
        tanh = new TANH();

        for(auto& p : l_up->parameters()) params_.push_back(p);
        for(auto& p : l_down->parameters()) params_.push_back(p);
    }

    ~MLP() {
        delete l_up;
        delete l_down;
        delete tanh;
    }

    Value operator()(Value x) override {
        x = (*l_up)(x);
        x = (*tanh)(x);
        x = (*l_down)(x);
        return x;
    }

    Linear* l_up;
    Linear* l_down;
    TANH* tanh;
};

class GPT : public Module {
public:
    GPT(GPTConfig config) : config(config) {
        // Initialize embedding weights on CPU, will be moved to GPU
        std::vector<float> wte_data(config.vocab_size * config.n_embd);
        std::vector<float> wpe_data(config.context_length * config.n_embd);
        
        std::mt19937 rng(1337);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for(auto& w : wte_data) w = dist(rng);
        for(auto& w : wpe_data) w = dist(rng);
        
        // CPU cache for fast embedding lookup (avoid repeated GPU->CPU transfers)
        wte_cpu_cache = Tensor(Shape{{static_cast<int64_t>(config.vocab_size), static_cast<int64_t>(config.n_embd)}}, 
                               Dtype::Float32, Device::CPU);
        std::copy(wte_data.begin(), wte_data.end(), wte_cpu_cache.data<float>());
        
        // Token embedding on GPU - requires_grad=true for training
        // Used for weight-tied output projection (matmul with logits)
        Tensor wte_t = Tensor(Shape{{static_cast<int64_t>(config.vocab_size), static_cast<int64_t>(config.n_embd)}}, 
                              Dtype::Float32, Device::CUDA, true);  // requires_grad=true, on GPU
        // Copy data via CPU tensor
        Tensor wte_cpu = Tensor(Shape{{static_cast<int64_t>(config.vocab_size), static_cast<int64_t>(config.n_embd)}}, 
                                Dtype::Float32, Device::CPU);
        std::copy(wte_data.begin(), wte_data.end(), wte_cpu.data<float>());
        cudaMemcpy(wte_t.data<float>(), wte_cpu.data<float>(), 
                   config.vocab_size * config.n_embd * sizeof(float), cudaMemcpyHostToDevice);
        wte = make_tensor(wte_t, "wte");
        params_.push_back(wte);
        
        // CPU cache for positional embedding
        wpe_cpu_cache = Tensor(Shape{{static_cast<int64_t>(config.context_length), static_cast<int64_t>(config.n_embd)}}, 
                               Dtype::Float32, Device::CPU);
        std::copy(wpe_data.begin(), wpe_data.end(), wpe_cpu_cache.data<float>());
        
        // Positional embedding on GPU - requires_grad=true for training
        Tensor wpe_t = Tensor(Shape{{static_cast<int64_t>(config.context_length), static_cast<int64_t>(config.n_embd)}}, 
                              Dtype::Float32, Device::CUDA, true);  // requires_grad=true, on GPU
        Tensor wpe_cpu = Tensor(Shape{{static_cast<int64_t>(config.context_length), static_cast<int64_t>(config.n_embd)}}, 
                                Dtype::Float32, Device::CPU);
        std::copy(wpe_data.begin(), wpe_data.end(), wpe_cpu.data<float>());
        cudaMemcpy(wpe_t.data<float>(), wpe_cpu.data<float>(), 
                   config.context_length * config.n_embd * sizeof(float), cudaMemcpyHostToDevice);
        wpe = make_tensor(wpe_t, "wpe");
        params_.push_back(wpe);
        
        mlp = new MLP(config);
        // Weight tying: use wte for output projection (no separate finall layer)

        for(auto& p : mlp->parameters()) params_.push_back(p);
    }

    ~GPT() {
        delete mlp;
    }

    Value operator()(Value x) override {
        // Not used
        return Value();
    }

    // Index-based embedding lookup - no one-hot encoding needed
    // Input: ids (B, T) as uint16, weight Tensor (V, C)
    // Output: Tensor (B*T, C) with embeddings looked up by index
    // Also stores indices in last_*_indices for gradient accumulation
    Tensor embed_lookup(const Tensor& ids, const Tensor& weight, int64_t B, int64_t T) {
        int64_t V = weight.shape().dims[0];
        int64_t C = weight.shape().dims[1];
        int64_t N = B * T;
        
        // Get weight and ids on CPU for lookup
        Tensor weight_cpu = (weight.device().device != Device::CPU) ? weight.to_cpu() : weight;
        Tensor ids_cpu = (ids.device().device != Device::CPU) ? ids.to_cpu() : ids;
        
        const float* w_ptr = weight_cpu.data<float>();
        const uint16_t* ids_ptr = ids_cpu.data<uint16_t>();
        
        // Create output tensor and perform lookup
        Tensor output = Tensor(Shape{{N, C}}, Dtype::Float32, Device::CPU);
        float* out_ptr = output.data<float>();
        
        for (int64_t i = 0; i < N; ++i) {
            int idx = static_cast<int>(ids_ptr[i]);
            if (idx >= 0 && idx < V) {
                // Copy row idx from weight to output row i
                std::copy(w_ptr + idx * C, w_ptr + (idx + 1) * C, out_ptr + i * C);
            } else {
                // Out of bounds - zero fill
                std::fill(out_ptr + i * C, out_ptr + (i + 1) * C, 0.0f);
            }
        }
        
        return output.to(Device::CUDA);
    }
    
    
    // Store last used indices and embedding Values for gradient accumulation
    Tensor last_tok_ids;
    Tensor last_pos_ids;
    Value last_tok_emb;  // For gradient retrieval after backward
    Value last_pos_emb;  // For gradient retrieval after backward

    std::pair<Value, Value> forward(const Tensor& input_ids, const Tensor& target_ids) {
        int64_t B = input_ids.shape().dims[0];
        int64_t T = input_ids.shape().dims[1];
        int64_t C = config.n_embd;
        int64_t N = B * T;

        // Store indices for gradient accumulation after backward
        last_tok_ids = input_ids;
        
        // Get token embeddings via index lookup (no one-hot)
        // Use CPU cache to avoid GPU->CPU copy of weights
        Tensor tok_emb_t = embed_lookup(input_ids, wte_cpu_cache, B, T);  // (B*T, C) GPU tensor
        tok_emb_t.set_requires_grad(true);  // Enable gradient tracking
        Value tok_emb = make_tensor(tok_emb_t, "tok_emb");
        last_tok_emb = tok_emb;  // Store for gradient accumulation
        
        // Get positional embeddings
        std::vector<uint16_t> pos_data(N);
        for(int b = 0; b < B; ++b) {
            for(int t = 0; t < T; ++t) {
                pos_data[b * T + t] = t;
            }
        }
        Tensor pos_ids = Tensor(Shape{{B, T}}, Dtype::UInt16, Device::CPU);
        std::copy(pos_data.begin(), pos_data.end(), pos_ids.data<uint16_t>());
        last_pos_ids = pos_ids;
        
        // Use CPU cache for fast positional embedding lookup
        Tensor pos_emb_t = embed_lookup(pos_ids, wpe_cpu_cache, B, T);  // (B*T, C)
        pos_emb_t.set_requires_grad(true);  // Enable gradient tracking
        Value pos_emb = make_tensor(pos_emb_t, "pos_emb");
        last_pos_emb = pos_emb;  // Store for gradient accumulation
        

        
        // Add embeddings - stays 2D (B*T, C)
        Value x = ag::add(tok_emb, pos_emb);
        


        // MLP blocks with residual connections
        // Works with 2D (B*T, C) - Linear handles this correctly
        for(int i = 0; i < config.n_layers; ++i) {
            Value residual = x;
            Value m = (*mlp)(x);
            x = ag::add(residual, m);
            

        }

        // wte shape: (V, C), so wte.t() is (C, V)
        // x shape: (B*T, C), output: (B*T, V)
        

        
        Value logits = ag::matmul(x, ag::transpose(wte));

        // Sparse Cross-entropy loss matching PyTorch's F.cross_entropy(logits, targets)
        // Uses softmax from package, sparse indexing with target indices (no one-hot)
        // Formula: loss = -mean(log(softmax(logits)[i, target[i]]))
        // Gradient: d_loss/d_logits[i,j] = (softmax[i,j] - (j == target[i] ? 1 : 0)) / N
        float loss_val = 0.0f;
        if (target_ids.numel() > 0) {
            int64_t V = config.vocab_size;
            
            // Step 1: Compute softmax using package function
            Tensor softmax_gpu = OwnTensor::mlp_forward::softmax(logits.val(), -1);  // (N, V)
            
            // Step 2: Move to CPU for sparse loss computation with index targets
            Tensor t_cpu = (target_ids.device().device != Device::CPU) ? target_ids.to_cpu() : target_ids;
            Tensor softmax_cpu = softmax_gpu.to_cpu();
            
            const float* sm_ptr = softmax_cpu.data<float>();
            const uint16_t* t_ptr = t_cpu.data<uint16_t>();
            
            // Compute sparse loss: -mean(log(softmax[i, target[i]]))
            float loss_sum = 0.0f;
            for (int64_t i = 0; i < N; ++i) {
                int label = static_cast<int>(t_ptr[i]);
                if (label >= 0 && label < V) {
                    loss_sum -= std::log(sm_ptr[i * V + label] + 1e-10f);
                }
            }
            loss_val = loss_sum / static_cast<float>(N);
            
            // Step 3: Compute sparse gradient: (softmax - sparse_onehot) / N
            // Only subtract 1 at the target index, rest stays as softmax/N
            Tensor grad_cpu = Tensor(Shape{{N, V}}, Dtype::Float32, Device::CPU);
            float* grad_ptr = grad_cpu.data<float>();
            float inv_N = 1.0f / static_cast<float>(N);
            
            // Copy softmax * inv_N to gradient
            for (int64_t i = 0; i < N * V; ++i) {
                grad_ptr[i] = sm_ptr[i] * inv_N;
            }
            
            // Subtract 1/N at target indices (sparse update)
            for (int64_t i = 0; i < N; ++i) {
                int label = static_cast<int>(t_ptr[i]);
                if (label >= 0 && label < V) {
                    grad_ptr[i * V + label] -= inv_N;
                }
            }
            
            // Move gradient back to GPU
            Tensor grad_gpu = grad_cpu.to(Device::CUDA);
            logits.node->grad = grad_gpu;
        }

        // Create loss Value manually (loss is a scalar, gradient was set on logits)
        Tensor loss_t = Tensor(Shape{{1}}, Dtype::Float32, Device::CUDA);
        loss_t.fill(loss_val);
        Value loss = make_tensor(loss_t, "loss");
        
        // Connect logits to the graph for backward propagation
        loss.node->inputs = {logits.node};
        logits.node->child_grad_count++;

        return {logits, loss};
    }

    int64_t count_parameters() const {
        int64_t total = 0;
        for (const auto& p : params_) {
            total += p.val().numel();
        }
        return total;
    }
    
    // Accumulate embedding gradients from tok_emb/pos_emb back to wte/wpe
    // This is needed because embedding lookup bypasses autodiff graph
    void accumulate_embedding_grads() {
        int64_t V_tok = config.vocab_size;
        int64_t V_pos = config.context_length;
        int64_t C = config.n_embd;
        
        // Accumulate token embedding gradients to wte
        if (last_tok_emb.node && last_tok_emb.node->grad.numel() > 0 && last_tok_ids.numel() > 0) {
            Tensor grad = last_tok_emb.node->grad;
            Tensor grad_cpu = (grad.device().device != Device::CPU) ? grad.to_cpu() : grad;
            Tensor ids_cpu = (last_tok_ids.device().device != Device::CPU) ? last_tok_ids.to_cpu() : last_tok_ids;
            
            // Ensure wte has gradient tensor
            if (wte.node->grad.numel() == 0) {
                wte.node->grad = Tensor::zeros(wte.val().shape(), options(wte.val()));
            }
            
            // Scatter-add on CPU, then copy back
            Tensor wte_grad_cpu = wte.node->grad.to_cpu();
            const float* grad_ptr = grad_cpu.data<float>();
            const uint16_t* ids_ptr = ids_cpu.data<uint16_t>();
            float* wte_grad_ptr = wte_grad_cpu.data<float>();
            
            int64_t N = last_tok_ids.numel();
            for (int64_t i = 0; i < N; ++i) {
                int idx = static_cast<int>(ids_ptr[i]);
                if (idx >= 0 && idx < V_tok) {
                    for (int64_t c = 0; c < C; ++c) {
                        wte_grad_ptr[idx * C + c] += grad_ptr[i * C + c];
                    }
                }
            }
            
            // Copy back to GPU
            cudaMemcpy(wte.node->grad.data<float>(), wte_grad_ptr, 
                       V_tok * C * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        // Accumulate positional embedding gradients to wpe
        if (last_pos_emb.node && last_pos_emb.node->grad.numel() > 0 && last_pos_ids.numel() > 0) {
            Tensor grad = last_pos_emb.node->grad;
            Tensor grad_cpu = (grad.device().device != Device::CPU) ? grad.to_cpu() : grad;
            Tensor ids_cpu = (last_pos_ids.device().device != Device::CPU) ? last_pos_ids.to_cpu() : last_pos_ids;
            
            // Ensure wpe has gradient tensor
            if (wpe.node->grad.numel() == 0) {
                wpe.node->grad = Tensor::zeros(wpe.val().shape(), options(wpe.val()));
            }
            
            // Scatter-add on CPU, then copy back
            Tensor wpe_grad_cpu = wpe.node->grad.to_cpu();
            const float* grad_ptr = grad_cpu.data<float>();
            const uint16_t* ids_ptr = ids_cpu.data<uint16_t>();
            float* wpe_grad_ptr = wpe_grad_cpu.data<float>();
            
            int64_t N = last_pos_ids.numel();
            for (int64_t i = 0; i < N; ++i) {
                int idx = static_cast<int>(ids_ptr[i]);
                if (idx >= 0 && idx < V_pos) {
                    for (int64_t c = 0; c < C; ++c) {
                        wpe_grad_ptr[idx * C + c] += grad_ptr[i * C + c];
                    }
                }
            }
            
            // Copy back to GPU
            cudaMemcpy(wpe.node->grad.data<float>(), wpe_grad_ptr, 
                       V_pos * C * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    
    // Sync CPU embedding caches from GPU weights after optimizer step
    // This is CRITICAL: without this, embed_lookup uses stale weights
    // while output projection uses updated weights, causing loss divergence
    void sync_embedding_caches() {
        // Sync token embedding cache
        cudaMemcpy(wte_cpu_cache.data<float>(), wte.val().data<float>(), 
                   config.vocab_size * config.n_embd * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Sync positional embedding cache
        cudaMemcpy(wpe_cpu_cache.data<float>(), wpe.val().data<float>(), 
                   config.context_length * config.n_embd * sizeof(float), cudaMemcpyDeviceToHost);
    }

    GPTConfig config;
    Value wte, wpe;  // Embedding weights (wte also used for output via weight tying)
    Tensor wte_cpu_cache, wpe_cpu_cache;  // CPU cache for fast embedding lookup
    MLP* mlp;
};

// ============================================================================
// Learning Rate Schedule
// ============================================================================

float get_lr(int step, int warmup_steps, int max_steps, float max_lr, float min_lr) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) / static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

// ============================================================================
// Main Training Loop
// ============================================================================

int main() {
    try {
        std::cout << "===== GPT-2 MLP Training  =====\n";
        
        GPTConfig config;
        const int global_batch = 16384;
        const int B = 2;   // Reduced to 2 to prevent OOM (was 4)
        const int T = 1024;
        const std::string data_root = "/home/blu-bridge006/Desktop/cgadimpl/Data";
        
        cudaSetDevice(1);  // GPU 1 has more free memory
        
        std::cout << "Initializing model..." << std::endl;
        GPT model(config);
        model.to(DeviceIndex(Device::CUDA));
        

        
        int64_t num_params = model.count_parameters();
        std::cout << "Number of parameters: " << num_params << std::endl;
        
        std::cout << "Initializing dataloaders..." << std::endl;
        DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, true);
        DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, true);
        
        std::vector<Value> all_params = model.parameters();
        Adam optimizer(all_params, config.max_lr, 0.9f, 0.95f, 1e-8f);
        

        
        // Calculate gradient accumulation steps
        const int micro_batch_tokens = B * T;  // Tokens per micro-batch
        const int grad_accum_steps = global_batch / micro_batch_tokens;
        std::cout << "Gradient accumulation steps: " << grad_accum_steps << std::endl;
        std::cout << "Micro-batch size: " << B << " x " << T << " = " << micro_batch_tokens << " tokens" << std::endl;
        std::cout << "Global batch size: " << global_batch << " tokens\n" << std::endl;
        
        // Open log file for CSV output
        std::ofstream log_file(data_root + "/training_log3.csv");
        log_file << "step,train_loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec" << std::endl;
        std::cout << "Logging to: " << data_root << "/training_log3.csv" << std::endl;
        
        for (int step = 0; step < config.max_steps; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bool last_step = (step == config.max_steps - 1);
            
            // Validation
            float current_val_loss = -1.0f;
            if (step % 50 == 0 || last_step) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                const int val_loss_steps = 5;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    auto result = model.forward(batch.input, batch.target);
                    Value loss = result.second;
                    
                    Tensor l = loss.val().to_cpu();
                    val_loss_accum += l.data<float>()[0] / val_loss_steps;
                    
                    // Free computation graph memory
                    ag::detach_graph(loss);
                }
                
                current_val_loss = val_loss_accum;
                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
            }
            
            // Gradient accumulation training step
            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                Batch batch = train_loader.next_batch();
                auto result = model.forward(batch.input, batch.target);
                Value loss = result.second;
                
                // Backward accumulates gradients (sum, not mean)
                ag::backward(loss);
                
                // NOTE: We do NOT call accumulate_embedding_grads() here because:
                // - wte already receives gradients from output projection via weight tying
                // - Adding tok_emb gradients would double-count and cause gradient explosion
                // - wpe gradients from pos_emb are small and not critical for training
                
                // Accumulate loss for logging
                Tensor l = loss.val().to_cpu();
                loss_accum += l.data<float>()[0] / grad_accum_steps;
                
                // Free computation graph memory (critical for memory efficiency!)
                ag::detach_graph(loss);
            }
            
            // Scale gradients by 1/grad_accum_steps to get mean gradient
            for (auto& p : all_params) {
                if (p.node->requires_grad() && p.node->grad.numel() > 0) {
                    p.node->grad *= (1.0f / static_cast<float>(grad_accum_steps));
                }
            }
            
            // Clip gradients after accumulation
            float norm = ag::clip_grad_norm_(all_params, 1.0f);
            
            // Update learning rate
            float lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr);
            optimizer.set_alpha(lr);
            
            // Optimizer step (once per global batch)
            optimizer.step();
            
            // OPTIMIZED: Sync CPU embedding caches from updated GPU weights periodically
            // Embeddings change slowly, so syncing every 10 steps is sufficient
            // This reduces CPU-GPU memory transfer overhead and fragmentation
            if (step % 10 == 0 || last_step) {
                model.sync_embedding_caches();
            }
            
            cudaDeviceSynchronize();
            
            // Periodic CUDA memory cache clearing to prevent fragmentation
            // This is critical for long training runs to avoid OOM
            if (step % 10 == 0) {
                // Clear any cached allocations
                cudaDeviceSynchronize();
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
            int tokens_processed = global_batch;
            double tokens_per_sec = tokens_processed / dt;
            
            std::cout << "step " << std::setw(5) << step 
                      << " | loss: " << std::fixed << std::setprecision(6) << loss_accum
                      << " | lr " << std::scientific << std::setprecision(4) << lr
                      << " | norm: " << std::fixed << std::setprecision(4) << norm
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec
                      << std::endl;
            
            // Log
            log_file << step << "," 
                     << std::fixed << std::setprecision(6) << loss_accum << ","
                     << ((step % 50 == 0) ? std::to_string(current_val_loss) : "") << ","
                     << std::scientific << std::setprecision(6) << lr << ","
                     << std::fixed << std::setprecision(4) << norm << ","
                     << std::fixed << std::setprecision(2) << (dt * 1000) << ","
                     << std::fixed << std::setprecision(2) << tokens_per_sec
                     << std::endl;
            log_file.flush();  // Flush to ensure data is written
        }
        
        log_file.close();
        std::cout << "\nTraining complete! Log saved to: " << data_root << "/training_log3.csv" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
