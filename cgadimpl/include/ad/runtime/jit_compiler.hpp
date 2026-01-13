#pragma once
#include "ad/core/graph.hpp"
#include "ad/runtime/cuda_graphs.hpp"
#include "ad/core/mlir_emitter.hpp"
#include <vector>
#include <variant>
#include <memory>
#include <string>

namespace ag::jit {

// ===================================================================
// JIT Compiler Interface
// ===================================================================

struct Compiled {
    struct Impl;
    std::shared_ptr<Impl> p;

    // MLIR data
    std::string mlir_source;
    std::shared_ptr<void> mlir_module;
    std::string mlir_module_str;
    void* compiled_func = nullptr;
    
    // AOT compiled function data (populated during compile())
    void* compiled_func = nullptr;      // Function pointer to _mlir_ciface_main
    void* dl_handle = nullptr;          // Handle from dlopen (for cleanup)
    
    // Execute the compiled plan
    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const;

    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             std::vector<Tensor>& outs) const;

    const std::string& getMLIRSource() const;
    void* getMLIRModule() const;
};

struct CompileOptions {
    bool use_cuda_graph = false;
    bool include_backward = false;
    // ... optimization flags ...
};

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions& opts = {});

} // namespace ag::jit
