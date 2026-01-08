#include "ad/runtime/jit_compiler.hpp"
#include "ad/ops/nodeops.hpp" 
#include "TensorLib.h"
#include "ad/core/mlir_emitter.hpp"
#include "Compiler/API/NovaCompilerAPI.h"
#include "mlir/IR/BuiltinOps.h"
#include <unordered_map>
#include <variant>
#include <iostream>
#include <cassert>
#include <sstream>

#include "Runtime/Executor/ExecutionEngine.h"
#include "Runtime/Kernels/KernelRegistration.h"
#include <dlfcn.h>
#include <fstream>
#include <unistd.h>

namespace ag::jit {

using ag::Op;
using ag::Node;

// ===================================================================
// JIT Compiler Implementation
// ===================================================================

struct Compiled::Impl {
    Plan plan;

    // --- helpers for replay ---
    static const Tensor& as_ref(const Arg& a,
                                const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& params,
                                const std::vector<Tensor>& slots,
                                Tensor& tmp) {
        if (std::holds_alternative<ArgInput>(a))  return *inputs[std::get<ArgInput>(a).idx];
        if (std::holds_alternative<ArgParam>(a))  return *params[std::get<ArgParam>(a).idx];
        if (std::holds_alternative<ArgSlot>(a))   return slots[std::get<ArgSlot>(a).slot];
        // literal: copy into tmp to return a ref
        const Tensor& lit = std::get<ArgLit>(a).t;
        tmp = lit;
        return tmp;
    }

    static Tensor apply(Op op, const std::vector<const Tensor*>& a) {
        // a.size() equals op_arity(op), except literals we materialized as tensors
        switch(op){
            case Op::Add:        return *a[0] + *a[1];
            case Op::Sub:        return *a[0] - *a[1];
            case Op::Mul:        return *a[0] * *a[1];

            // Unary operators now use the free functions from the OwnTensor namespace.
            case Op::Transpose:  return a[0]->transpose(-2, -1);
            case Op::Relu:     { cudaStream_t stream = (cudaStream_t)ag::current_stream(); return (*a[0] + OwnTensor::abs(*a[0], stream)) * 0.5f;}
            case Op::Exp:        return OwnTensor::exp(*a[0]);
            case Op::Log:        return OwnTensor::log(*a[0]);
            case Op::Tanh:       return OwnTensor::tanh(*a[0]);
            
            case Op::MatMul:     return OwnTensor::matmul(*a[0], *a[1]);

            // Reductions need to be updated to the new API
            case Op::Sum: return OwnTensor::reduce_sum(*a[0]);
            case Op::RowSum: return OwnTensor::reduce_sum(*a[0], {1}, true);
            case Op::RowMax: return OwnTensor::reduce_max(*a[0], {1}, true);
            case Op::MeanAll: return OwnTensor::reduce_mean(*a[0]);

            case Op::Leaf: default: {
                // Shouldn't get called for Leaf
                assert(false && "apply(): unexpected op");
                return *a[0];
            }
        }
    }

    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const {
        if (!plan.sig.matches(inputs, params)) return false;

        std::vector<Tensor> slots(plan.num_slots);
        
        for (const Step& st : plan.steps) {
            if (st.out_slot >= 0) {
                slots[st.out_slot] = Tensor(OwnTensor::Shape{st.out_meta.shape}, st.out_meta.dtype, st.out_meta.device, false);
            }
        }

        // Execute
        for (const Step& st : plan.steps) {
            std::vector<const Tensor*> args; args.reserve(st.args.size());
            
            Tensor tmp{OwnTensor::Shape{}, OwnTensor::TensorOptions{}}; 
            
            std::vector<Tensor> tmp_keep; tmp_keep.reserve(st.args.size());
            for (const Arg& a : st.args) {
                if (std::holds_alternative<ArgLit>(a)) {
                    tmp_keep.emplace_back(std::get<ArgLit>(a).t);
                    args.push_back(&tmp_keep.back());
                } else {
                    args.push_back(&as_ref(a, inputs, params, slots, tmp));
                }
            }
            Tensor y = apply(st.op, args);
            slots[st.out_slot] = std::move(y);
        }

        out = slots[plan.out_slot];
        return true;
    }
};

static bool is_in(const std::unordered_map<Node*,int>& m, Node* n){ return m.find(n)!=m.end(); }

// --- Helpers for string-based MLIR emission (Fallback) ---
static std::string dtypeToMLIR(Dtype dt) {
    switch (dt) {
        case OwnTensor::Dtype::Float32:  return "f32";
        case OwnTensor::Dtype::Float16:  return "f16";
        case OwnTensor::Dtype::Bfloat16: return "bf16";
        case OwnTensor::Dtype::Int32:    return "i32";
        case OwnTensor::Dtype::Int64:    return "i64";
        default:                        return "unknown";
    }
}

static std::string shapeToMLIR(const std::vector<int64_t>& shape) {
    std::string s;
    for (int64_t dim : shape) {
        s += std::to_string(dim) + "x";
    }
    return s;
}

static std::string opToNovaOp(Op op) {
    switch (op) {
        case Op::Add:       return "nova.add";
        case Op::Mul:       return "nova.mul";
        case Op::MatMul:    return "nova.matmul"; 
        case Op::Sum:       return "nova.reduce<sum>";
        case Op::MeanAll:   return "nova.reduce<mean>";
        default:            return "nova.unknown_op";
    }
}

static std::string emitMLIR(const Plan& plan) {
    std::stringstream ss;
    ss << "func.func @main(";
    size_t arg_idx_counter = 0;

    auto print_arg_meta = [&](const std::vector<TensorMetadata>& metas) {
        for (size_t i = 0; i < metas.size(); ++i) {
            const auto& meta = metas[i];
            ss << "%arg" << arg_idx_counter++ << ": tensor<" 
               << shapeToMLIR(meta.shape) << dtypeToMLIR(meta.dtype) << ">";
            if (i < metas.size() - 1 || !plan.sig.param_meta.empty()) ss << "";
        }
    };

    print_arg_meta(plan.sig.in_meta);
    ss << ") -> tensor<" 
       << shapeToMLIR(plan.steps.back().out_meta.shape) 
       << dtypeToMLIR(plan.steps.back().out_meta.dtype) << "> {\n";

    std::unordered_map<int, std::string> slot_to_var_name;
    std::unordered_map<int, TensorMetadata> slot_to_meta;

    for (const auto& st : plan.steps) {
        slot_to_meta[st.out_slot] = st.out_meta;
    }

    for (size_t i = 0; i < plan.steps.size(); ++i) {
        const auto& st = plan.steps[i];
        std::string result_var = "%v" + std::to_string(i);
        slot_to_var_name[st.out_slot] = result_var;
        ss << "  " << result_var << " = " << opToNovaOp(st.op) << " ";
        std::vector<std::string> arg_names;
        std::vector<std::string> arg_types;

        for (const auto& arg : st.args) {
            std::visit([&](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, ArgInput> || std::is_same_v<T, ArgParam>) {
                    int arg_idx = a.idx; 
                    const auto& meta = (std::is_same_v<T, ArgInput>) ? plan.sig.in_meta[arg_idx] : plan.sig.param_meta[arg_idx];
                    arg_names.push_back("%arg" + std::to_string(arg_idx));
                    arg_types.push_back("tensor<" + shapeToMLIR(meta.shape) + dtypeToMLIR(meta.dtype) + ">");
                } else if constexpr (std::is_same_v<T, ArgSlot>) {
                    arg_names.push_back(slot_to_var_name.at(a.slot));
                    const auto& meta = slot_to_meta.at(a.slot);
                    arg_types.push_back("tensor<" + shapeToMLIR(meta.shape) + dtypeToMLIR(meta.dtype) + ">");
                } else if constexpr (std::is_same_v<T, ArgLit>) {
                    arg_names.push_back("const_lit"); 
                    arg_types.push_back("tensor<f32>");
                }
            }, arg);
        }

        for (size_t j = 0; j < arg_names.size(); ++j) {
            ss << arg_names[j];
            if (j < arg_names.size() - 1) ss << ", ";
        }
        ss << ": ";
        for (size_t j = 0; j < arg_types.size(); ++j) {
            ss << arg_types[j];
            if (j < arg_types.size() - 1) ss << ", ";
        }
        ss << "\n";
    }

    std::string return_var = slot_to_var_name.at(plan.out_slot);
    const auto& return_meta = plan.steps.back().out_meta;
    auto return_shape = return_meta.shape;
    
    if ((plan.steps.back().op == Op::Sum || plan.steps.back().op == Op::MeanAll) && 
        return_shape.size() == 1 && return_shape[0] == 1) {
        return_shape = {};
    }

    ss << "  return " << return_var << " : tensor<"
       << shapeToMLIR(return_shape) 
       << dtypeToMLIR(return_meta.dtype) << ">\n";
    ss << "}\n";
    return ss.str();
}

// Forward declarations for AOT compilation helpers
void* compileAndLoad(const std::string& mlir_source);
void storeAOTContext(void* func_ptr, const Plan& plan, const std::string& mlir_source);

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions&) {
    std::unordered_map<Node*,int> in_ix, par_ix;
    for (size_t i = 0; i < inputs.size(); ++i) in_ix[inputs[i].node.get()] = i;
    for (size_t i = 0; i < params.size(); ++i) par_ix[params[i].node.get()] = i;

    Plan plan;
    plan.sig.in_meta.reserve(inputs.size());
    for (const auto& v: inputs) {
        plan.sig.in_meta.push_back({v.shape(), v.val().dtype(), v.val().device()});
    }
    plan.sig.param_meta.reserve(params.size());
    for (const auto& v: params) {
        plan.sig.param_meta.push_back({v.shape(), v.val().dtype(), v.val().device()});
    }

    auto order = topo_from(output.node.get());
    std::unordered_map<Node*,int> slot_of;
    slot_of.reserve(order.size());

    for (Node* n : order) {
        if (n->op == Op::Leaf) continue;
        Step st;
        st.op = n->op;
        st.out_meta = {n->shape(), n->value.dtype(), n->value.device()};
        st.out_slot = plan.num_slots++;
        slot_of[n] = st.out_slot;

        st.args.reserve(n->inputs.size());
        for (auto& pin : n->inputs) {
            Node* p = pin.get();
            if (p->op == Op::Leaf) {
                if (is_in(in_ix, p))        st.args.push_back(ArgInput{ in_ix[p] });
                else if (is_in(par_ix, p))  st.args.push_back(ArgParam{ par_ix[p] });
                else                        st.args.push_back(ArgLit{ p->value });
            } else {
                st.args.push_back(ArgSlot{ slot_of.at(p) });
            }
        }
        plan.steps.push_back(std::move(st));
    }
    plan.out_slot = slot_of.at(output.node.get());

    std::string generated_mlir_opbuilder;
    mlir::OwningOpRef<mlir::ModuleOp> in_memory_module;
    std::shared_ptr<mlir::MLIRContext> context;
    
    try {
        MLIREmitter emitter;
        context = emitter.getContext();
        auto [module, mlirStr] = emitter.emitModule(plan);
        generated_mlir_opbuilder = mlirStr;
        in_memory_module = std::move(module);
        std::cout << "\n=== MLIR Generated via OpBuilder ===\n" << generated_mlir_opbuilder << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: MLIR OpBuilder emission failed: " << e.what() << "\n";
    }

    std::string generated_mlir_string = emitMLIR(plan);
    if (generated_mlir_opbuilder.empty()) {
        std::cout << "\n=== MLIR Generated via String (Fallback) ===\n" << generated_mlir_string << std::endl;
    }

    Compiled c;
    c.p = std::make_shared<Compiled::Impl>();
    c.p->plan = std::move(plan);
    
    // Store the OpBuilder MLIR (correct scalar representation) if available,
    // otherwise fall back to string-based MLIR
    c.mlir_source = generated_mlir_opbuilder.empty() ? generated_mlir_string : generated_mlir_opbuilder;
    
    if (in_memory_module) {
        try {
            mlir::nova::NovaCompilerAPI compiler;
            mlir::nova::CompilerOptions options;
            options.runFullPipeline = true;
            auto compileResult = compiler.compileString(generated_mlir_opbuilder, "", options);
            if (compileResult.success) {
                generated_mlir_opbuilder = compileResult.output;
                std::cout << "\n=== Optimized MLIR Generated via NovaCompilerAPI ===\n"  << std::endl;
            } else {
                std::cerr << "Warning: NovaCompilerAPI pipeline failed: " << compileResult.errorMessage << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: NovaCompilerAPI integration failed: " << e.what() << "\n";
        }

        auto* module_ptr = new mlir::OwningOpRef<mlir::ModuleOp>(std::move(in_memory_module));
        c.mlir_module = std::shared_ptr<void>(module_ptr, [context](void* p) {
            delete static_cast<mlir::OwningOpRef<mlir::ModuleOp>*>(p);
        });
    }

    c.mlir_module_str = std::move(generated_mlir_opbuilder);
    
    // ===================================================================
    // NEW: Perform AOT Compilation Immediately (Eager Compilation)
    // ===================================================================
    if (!c.mlir_source.empty()) {
        std::cout << "\n[Compile] Starting AOT compilation...\n";
        void* func_ptr = compileAndLoad(c.mlir_source);
        
        if (func_ptr) {
            c.compiled_func = func_ptr;
            // Store the Plan metadata and original MLIR source for signature parsing
            storeAOTContext(func_ptr, c.p->plan, c.mlir_source);
            std::cout << "[Compile] AOT compilation successful! Function ready at " << func_ptr << "\n";
        } else {
            std::cerr << "[Compile] Warning: AOT compilation failed. Will fall back to interpreter.\n";
        }
    }
    
    return c;
}

// ===================================================================
// AOT Adapter Function
// ===================================================================

// Global storage for the compiled function and metadata
struct AOTContext {
    void* ciface_func = nullptr;
    Plan plan;
    std::string mlir_source;  // Store MLIR source for signature parsing
};

static AOTContext g_aot_context;

// Generic memref descriptor builder
// Returns a vector of bytes representing the memref descriptor
std::vector<char> buildMemRefDescriptor(Tensor* tensor, const TensorMetadata& meta) {
    size_t rank = meta.shape.size();
    
    size_t descriptor_size = sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * rank);
    std::vector<char> descriptor(descriptor_size, 0);
    
    char* ptr = descriptor.data();
    
    // Write allocated pointer
    *reinterpret_cast<void**>(ptr) = tensor->data<float>();
    ptr += sizeof(void*);
    
    // Write aligned pointer
    *reinterpret_cast<void**>(ptr) = tensor->data<float>();
    ptr += sizeof(void*);
    
    // Write offset
    *reinterpret_cast<int64_t*>(ptr) = 0;
    ptr += sizeof(int64_t);
    
    // Write sizes
    for (size_t i = 0; i < rank; ++i) {
        *reinterpret_cast<int64_t*>(ptr) = tensor->shape().dims[i];
        ptr += sizeof(int64_t);
    }
    
    // Write strides (row-major)
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        *reinterpret_cast<int64_t*>(ptr) = stride;
        ptr += sizeof(int64_t);
        stride *= tensor->shape().dims[i];
    }
    
    return descriptor;
}

// Parse MLIR function signature to extract result tensor shape
// Returns: vector of dimensions (empty for scalar)
std::vector<int64_t> parseMLIRResultShape(const std::string& mlir_source) {
    std::vector<int64_t> result_shape;
    
    // Find the function signature: "-> tensor<...>"
    size_t arrow_pos = mlir_source.find("-> tensor<");
    if (arrow_pos == std::string::npos) {
        std::cerr << "[ABIAdapter] Warning: Could not find result tensor in MLIR signature\n";
        return result_shape;  // Empty = scalar
    }
    
    size_t start = arrow_pos + 10;  // After "-> tensor<"
    size_t end = mlir_source.find(">", start);
    if (end == std::string::npos) {
        std::cerr << "[ABIAdapter] Warning: Malformed tensor signature\n";
        return result_shape;
    }
    
    std::string tensor_spec = mlir_source.substr(start, end - start);
    
    // Check if it's a scalar (just "f32" or similar type)
    if (tensor_spec.find('x') == std::string::npos) {
        // Scalar - no dimensions
        return result_shape;  // Empty vector
    }
    
    // Parse dimensions: "8x10xf32" -> [8, 10]
    size_t pos = 0;
    while (pos < tensor_spec.size()) {
        size_t x_pos = tensor_spec.find('x', pos);
        if (x_pos == std::string::npos) {
            // Last part is the type (e.g., "f32"), not a dimension
            break;
        }
        
        std::string dim_str = tensor_spec.substr(pos, x_pos - pos);
        try {
            int64_t dim = std::stoll(dim_str);
            result_shape.push_back(dim);
        } catch (...) {
            std::cerr << "[ABIAdapter] Warning: Failed to parse dimension: " << dim_str << "\n";
            break;
        }
        
        pos = x_pos + 1;
    }
    
    return result_shape;
}

extern "C" void* ABIAdapter(void** args) {
    if (!g_aot_context.ciface_func) {
        std::cerr << "[ABIAdapter] Error: No compiled function loaded\n";
        return nullptr;
    }
    
    const Plan& plan = g_aot_context.plan;
    size_t num_inputs = plan.sig.in_meta.size();
    size_t num_params = plan.sig.param_meta.size();
    size_t total_args = num_inputs + num_params;
    
    std::cout << "[ABIAdapter] Generic adapter executing with " 
              << num_inputs << " inputs, " << num_params << " params\n";
    
    // Build memref descriptors for all arguments
    std::vector<std::vector<char>> descriptors;
    descriptors.reserve(total_args);
    
    // Build descriptors for inputs
    for (size_t i = 0; i < num_inputs; ++i) {
        auto* tensor = static_cast<Tensor*>(args[i]);
        descriptors.push_back(buildMemRefDescriptor(tensor, plan.sig.in_meta[i]));
        
        std::cout << "[ABIAdapter]   Input[" << i << "]: shape=(";
        for (size_t j = 0; j < plan.sig.in_meta[i].shape.size(); ++j) {
            std::cout << plan.sig.in_meta[i].shape[j];
            if (j < plan.sig.in_meta[i].shape.size() - 1) std::cout << "x";
        }
        std::cout << ")\n";
    }
    
    // Build descriptors for params
    for (size_t i = 0; i < num_params; ++i) {
        auto* tensor = static_cast<Tensor*>(args[num_inputs + i]);
        descriptors.push_back(buildMemRefDescriptor(tensor, plan.sig.param_meta[i]));
        
        std::cout << "[ABIAdapter]   Param[" << i << "]: shape=(";
        for (size_t j = 0; j < plan.sig.param_meta[i].shape.size(); ++j) {
            std::cout << plan.sig.param_meta[i].shape[j];
            if (j < plan.sig.param_meta[i].shape.size() - 1) std::cout << "x";
        }
        std::cout << ")\n";
    }
    
    // Parse MLIR signature to get result shape
    std::vector<int64_t> result_shape = parseMLIRResultShape(g_aot_context.mlir_source);
    size_t result_rank = result_shape.size();
    
    std::cout << "[ABIAdapter] Result shape from MLIR: ";
    if (result_rank == 0) {
        std::cout << "scalar";
    } else {
        std::cout << "(";
        for (size_t i = 0; i < result_rank; ++i) {
            std::cout << result_shape[i];
            if (i < result_rank - 1) std::cout << "x";
        }
        std::cout << ")";
    }
    std::cout << "\n";
    
    // Allocate result descriptor based on parsed rank
    std::vector<char> result_descriptor;
    if (result_rank == 0) {
        // Scalar: 2 pointers + offset
        result_descriptor.resize(sizeof(void*) * 2 + sizeof(int64_t), 0);
    } else {
        // Non-scalar: 2 pointers + offset + rank sizes + rank strides
        size_t desc_size = sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * result_rank);
        result_descriptor.resize(desc_size, 0);
    }
    
    // Build array of descriptor pointers for the C interface call
    std::vector<void*> descriptor_ptrs;
    descriptor_ptrs.reserve(total_args + 1);
    descriptor_ptrs.push_back(result_descriptor.data());  // Result is first argument
    for (auto& desc : descriptors) {
        descriptor_ptrs.push_back(desc.data());
    }
    
    std::cout << "[ABIAdapter] Calling compiled function...\n";
    
    // Call the compiled function using variadic approach
    using CIfaceFunc1 = void (*)(void*);
    using CIfaceFunc2 = void (*)(void*, void*);
    using CIfaceFunc3 = void (*)(void*, void*, void*);
    using CIfaceFunc4 = void (*)(void*, void*, void*, void*);
    using CIfaceFunc5 = void (*)(void*, void*, void*, void*, void*);
    
    switch (descriptor_ptrs.size()) {
        case 1:
            reinterpret_cast<CIfaceFunc1>(g_aot_context.ciface_func)(descriptor_ptrs[0]);
            break;
        case 2:
            reinterpret_cast<CIfaceFunc2>(g_aot_context.ciface_func)(descriptor_ptrs[0], descriptor_ptrs[1]);
            break;
        case 3:
            reinterpret_cast<CIfaceFunc3>(g_aot_context.ciface_func)(descriptor_ptrs[0], descriptor_ptrs[1], descriptor_ptrs[2]);
            break;
        case 4:
            reinterpret_cast<CIfaceFunc4>(g_aot_context.ciface_func)(descriptor_ptrs[0], descriptor_ptrs[1], descriptor_ptrs[2], descriptor_ptrs[3]);
            break;
        case 5:
            reinterpret_cast<CIfaceFunc5>(g_aot_context.ciface_func)(descriptor_ptrs[0], descriptor_ptrs[1], descriptor_ptrs[2], descriptor_ptrs[3], descriptor_ptrs[4]);
            break;
        default:
            std::cerr << "[ABIAdapter] Error: Unsupported number of arguments: " << descriptor_ptrs.size() << "\n";
            return nullptr;
    }
    
    std::cout << "[ABIAdapter] Function returned\n";
    
    // Extract result from descriptor
    char* result_ptr = result_descriptor.data();
    void* allocated = *reinterpret_cast<void**>(result_ptr);
    void* aligned = *reinterpret_cast<void**>(result_ptr + sizeof(void*));
    int64_t offset = *reinterpret_cast<int64_t*>(result_ptr + sizeof(void*) * 2);
    
    if (!aligned) {
        std::cerr << "[ABIAdapter] Error: Invalid result from compiled function\n";
        return nullptr;
    }
    
    // Create output tensor based on result rank
    Tensor* out;
    if (result_rank == 0) {
        // Scalar result
        out = new Tensor(
            OwnTensor::Shape{{1}},
            OwnTensor::Dtype::Float32,
            OwnTensor::DeviceIndex(OwnTensor::Device::CPU),
            false
        );
        
        float* data_ptr = reinterpret_cast<float*>(aligned) + offset;
        out->data<float>()[0] = *data_ptr;
    } else {
        // Non-scalar result - extract actual sizes from descriptor
        int64_t* sizes_ptr = reinterpret_cast<int64_t*>(result_ptr + sizeof(void*) * 2 + sizeof(int64_t));
        std::vector<int64_t> actual_shape(sizes_ptr, sizes_ptr + result_rank);
        
        out = new Tensor(
            OwnTensor::Shape{actual_shape},
            OwnTensor::Dtype::Float32,
            OwnTensor::DeviceIndex(OwnTensor::Device::CPU),
            false
        );
        
        // Calculate total elements
        size_t total_elements = 1;
        for (auto dim : actual_shape) {
            total_elements *= dim;
        }
        
        // Copy data
        float* data_ptr = reinterpret_cast<float*>(aligned) + offset;
        std::memcpy(out->data<float>(), data_ptr, total_elements * sizeof(float));
    }
    
    std::free(allocated);
    
    std::cout << "[ABIAdapter] Successfully executed, result: ";
    if (result_rank == 0) {
        std::cout << "scalar value = " << out->data<float>()[0];
    } else {
        std::cout << "tensor shape (";
        for (size_t i = 0; i < result_rank; ++i) {
            std::cout << out->shape().dims[i];
            if (i < result_rank - 1) std::cout << "x";
        }
        std::cout << "), first value = " << out->data<float>()[0];
    }
    std::cout << "\n";
    
    return out;
}


// Wrapper Declaration (must be global)
extern "C" void* LegacyInterpWrapper(void**);

// Helper to compile MLIR string to Shared Object and Load it
void* compileAndLoad(const std::string& mlir_source) {
    std::string base_path = "/tmp/nova_jit_" + std::to_string(std::rand());
    std::string mlir_file = base_path + ".mlir";
    std::string obj_file = base_path + ".o";
    std::string so_file = base_path + ".so";
    
    // 1. Write MLIR to file
    {
        std::ofstream out(mlir_file);
        out << mlir_source;
        out.close();
    }
    
    std::cout << "[NovaAOT] Compiling " << mlir_file << " to object code...\n";
    
        // 2. Compile to Object File using SystemAPI
    #ifndef NOVA_OPT_BIN
 
        #define NOVA_OPT_BIN "nova-opt"
    #endif
        std::string nova_opt = NOVA_OPT_BIN;
        bool success = mlir::nova::NovaCompilerSystemAPI::compileToObject(mlir_file, obj_file, nova_opt);
    if (!success) {
        std::cerr << "[NovaAOT] Compilation failed.\n";
        return nullptr;
    }
    
    std::cout << "[NovaAOT] Linking " << obj_file << " to shared object...\n";
    // 3. Link to Shared Object (Required for dlopen)
    // We invoke gcc/ld to convert .o -> .so
    std::string link_cmd = "gcc -shared -fPIC -o " + so_file + " " + obj_file;
    if (system(link_cmd.c_str()) != 0) {
        std::cerr << "[NovaAOT] Linking failed: " << link_cmd << "\n";
        return nullptr;
    }
    
    std::cout << "[NovaAOT] Loading shared object " << so_file << "...\n";
    // 4. Load Shared Object
    void* handle = dlopen(so_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "[NovaAOT] dlopen failed: " << dlerror() << "\n";
        return nullptr;
    }
    
    // 5. Get Function Pointer
    // Default entry point for mlir-ciface is _mlir_ciface_main (if function is @main)
    // We assume the generated MLIR has @main
    void* func_ptr = dlsym(handle, "_mlir_ciface_main");
    if (!func_ptr) {
        // Fallback: try just "main" or "_mlir_main"
        func_ptr = dlsym(handle, "main");
    }
    
    if (!func_ptr) {
        std::cerr << "[NovaAOT] dlsym failed: Could not find _mlir_ciface_main or main\n";
        return nullptr;
    }
    
    std::cout << "[NovaAOT] Successfully loaded compiled kernel at " << func_ptr << "\n";
    return func_ptr;
}

// Helper to store AOT context (function + metadata)
void storeAOTContext(void* func_ptr, const Plan& plan, const std::string& mlir_source) {
    g_aot_context.ciface_func = func_ptr;
    g_aot_context.plan = plan;
    g_aot_context.mlir_source = mlir_source;
    std::cout << "[NovaAOT] Stored AOT context with " 
              << plan.sig.in_meta.size() << " inputs, "
              << plan.sig.param_meta.size() << " params\n";
}

// Wrapper to make the legacy 'run' look like a JIT compiled function
// Signature: void* func(void** args)
// args[0] = Compiled::Impl* (context)
// args[1] = std::vector<Tensor*>* (inputs)
// args[2] = std::vector<Tensor*>* (params)
// Returns: Tensor* (result)
extern "C" void* LegacyInterpWrapper(void** args) {
    auto* impl = static_cast<Compiled::Impl*>(args[0]);
    auto* inputs = static_cast<const std::vector<Tensor*>*>(args[1]);
    auto* params = static_cast<const std::vector<Tensor*>*>(args[2]);
    
    Tensor* out = new Tensor();
    bool success = impl->run(*inputs, *params, *out);
    if (!success) {
        delete out;
        return nullptr; // Signal failure?
    }
    return out;
}

bool Compiled::run(const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& params,
                   Tensor& out) const {
    
    // Check if we have a pre-compiled function from compile()
    if (!compiled_func) {
        // Fallback to interpreter if compilation failed or wasn't attempted
        std::cout << "[Run] No compiled function available, using interpreter...\n";
        return p->run(inputs, params, out);
    }

    // Execute the pre-compiled function via ABI adapter
    std::cout << "[Run] Executing pre-compiled function...\n";
    
    // Prepare arguments: all inputs first, then all params
    std::vector<void*> exec_inputs;
    for (auto* t : inputs) {
        exec_inputs.push_back(t);
    }
    for (auto* t : params) {
        exec_inputs.push_back(t);
    }
    
    // Call the ABI adapter directly 
    Tensor* result_tensor = static_cast<Tensor*>(ABIAdapter(exec_inputs.data()));
    
    if (!result_tensor) {
        std::cerr << "[Run] Error: ABI adapter returned null\n";
        return false;
    }

    // Move result to output
    out = std::move(*result_tensor);
    delete result_tensor;
    return true;
}

const std::string& Compiled::getMLIRSource() const {
    return mlir_source;
}

void* Compiled::getMLIRModule() const {
    if (mlir_module) {
        auto* module_ptr = static_cast<mlir::OwningOpRef<mlir::ModuleOp>*>(mlir_module.get());
        if (module_ptr && module_ptr->get()) {
            return module_ptr->get();
        }
    }
    return nullptr;
}

} // namespace ag::jit
