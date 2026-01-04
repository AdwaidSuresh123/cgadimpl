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
#include <cstring>
#include <cstdlib>

namespace ag::jit {

using ag::Op;
using ag::Node;

// ===================================================================
// ABI Adapter for MLIR C-Interface (Internal Helper)
// ===================================================================

/**
 * @brief dynamically constructs a MemRef descriptor in memory.
 * Layout:
 *   ptr allocated
 *   ptr aligned
 *   i64 offset
 *   i64 sizes[rank]
 *   i64 strides[rank]
 */
class MemRefBuilder {
public:
    MemRefBuilder(Tensor& t) : rank_(t.ndim()) {
        // Calculate size: 2 ptrs + 1 offset + 2*rank ints
        size_t size = sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * rank_);
        buffer_ = std::malloc(size);

        // Fill data
        intptr_t* ptr_buffer = static_cast<intptr_t*>(buffer_);
        int64_t* int_buffer = reinterpret_cast<int64_t*>(ptr_buffer + 2); // skip 2 ptrs

        // Pointers
        void* data_ptr = t.data();
        ptr_buffer[0] = reinterpret_cast<intptr_t>(data_ptr); // allocated
        ptr_buffer[1] = reinterpret_cast<intptr_t>(data_ptr); // aligned (assuming same)

        // Offset
        int_buffer[0] = (int64_t)t.storage_offset();

        // Sizes & Strides
        int64_t* sizes = &int_buffer[1];
        int64_t* strides = &int_buffer[1 + rank_];

        const auto& shape = t.shape().dims;
        // OwnTensor strides might need calculation if not stored readily,
        // but assuming t.stride() gives compatible vector
        const auto& stride_vec = t.stride().strides;

        // Fallback if strides empty (contiguous)
        if (stride_vec.empty()) {
            // Calculate contiguous strides
             int64_t s = 1;
             for (int i = rank_ - 1; i >= 0; --i) {
                 sizes[i] = shape[i];
                 strides[i] = s;
                 s *= shape[i];
             }
        } else {
            for (int i = 0; i < rank_; ++i) {
                sizes[i] = shape[i];
                strides[i] = stride_vec[i];
            }
        }
    }

    ~MemRefBuilder() {
        if (buffer_) std::free(buffer_);
    }

    void* get() const { return buffer_; }

private:
    int rank_;
    void* buffer_;
};


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
    c.mlir_source = std::move(generated_mlir_string);

    if (in_memory_module) {
        try {
            mlir::nova::NovaCompilerAPI compiler;
            mlir::nova::CompilerOptions options;
            options.runFullPipeline = true;
            auto compileResult = compiler.compileString(generated_mlir_opbuilder, "", options);
            if (compileResult.success) {
                // generated_mlir_opbuilder = compileResult.output;
                std::cout << "\n=== Optimized MLIR Generated via NovaCompilerAPI ===\n" << std::endl;
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
    return c;
}

// --- Bridge to Nova Runtime ---
// Includes moved to top of file to avoid namespace pollution
// See top of file for #include "Runtime/Executor/ExecutionEngine.h"

// --- Bridge to Nova Runtime (AOT Mode) ---
// Includes moved to top of file

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
    // Fallback if not defined by CMake (though it should be)
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

// ABI Adapter logic merged here
extern "C" void* JITABIAdapter(void** args) {
    // args[0]: Compiled::Impl*
    // args[1]: std::vector<Tensor*>* inputs
    // args[2]: std::vector<Tensor*>* params
    // args[3]: void* jit_func_ptr

    auto* impl = static_cast<Compiled::Impl*>(args[0]);
    auto* inputs_ptr = static_cast<const std::vector<Tensor*>*>(args[1]);
    auto* params_ptr = static_cast<const std::vector<Tensor*>*>(args[2]);
    void* jit_func_ptr = args[3];

    if (!jit_func_ptr) {
        std::cerr << "JITABIAdapter: Missing function pointer\n";
        return nullptr;
    }

    const auto& inputs = *inputs_ptr;
    const auto& params = *params_ptr;

    // 1. Prepare Output Tensor
    const auto& out_meta = impl->plan.steps.back().out_meta;
    Tensor* out = new Tensor(OwnTensor::Shape{out_meta.shape}, out_meta.dtype, out_meta.device, false);

    // 2. Build MemRef Descriptors
    std::vector<std::unique_ptr<MemRefBuilder>> descriptors;
    std::vector<void*> raw_args;

    // Output (SRet)
    {
        auto desc = std::make_unique<MemRefBuilder>(*out);
        raw_args.push_back(desc->get());
        descriptors.push_back(std::move(desc));
    }

    // Inputs
    for (Tensor* t : inputs) {
        auto desc = std::make_unique<MemRefBuilder>(*t);
        raw_args.push_back(desc->get());
        descriptors.push_back(std::move(desc));
    }

    // Params
    for (Tensor* t : params) {
        auto desc = std::make_unique<MemRefBuilder>(*t);
        raw_args.push_back(desc->get());
        descriptors.push_back(std::move(desc));
    }

    // 3. Call JIT Function (up to 8 args)
    using JITFunc = void (*)(void*, void*, void*, void*, void*, void*, void*, void*);
    auto func = reinterpret_cast<JITFunc>(jit_func_ptr);

    while(raw_args.size() < 8) raw_args.push_back(nullptr);

    func(raw_args[0], raw_args[1], raw_args[2], raw_args[3],
         raw_args[4], raw_args[5], raw_args[6], raw_args[7]);

    return out;
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

    // 1. Setup Runtime Engine
    // In a real app, HostContext should be a global singleton or passed in.
    static nova::runtime::HostContext host(4);
    nova::runtime::ExecutionEngine engine(&host);

    // 2. Get the JIT Function (Compile if not already compiled)
    // In a real implementation, we would cache 'jit_func_ptr' inside 'Compiled' class
    // For now, we compile every time (or simplistic static cache if we could)
    // But 'Compiled' object persists, so let's cache it there.
    // However, Compiled::run is const. We need mutable cache or const_cast.

    static void* cached_func = nullptr;
    if (!cached_func && !mlir_module_str.empty()) {
        cached_func = compileAndLoad(mlir_module_str);
    }

    if (!cached_func) {
        // Fallback to legacy behavior if compilation fails
         return p->run(inputs, params, out);
    }

    // 3. Build Execution Plan
    nova::runtime::RuntimeExecutionPlan plan;
    plan.output_task_id = 0;

    nova::runtime::AsyncTask task;
    task.task_id = 0;
    task.op_name = "jit.generated";
    task.device = nova::runtime::Device::CPU;

    // Use the ABI adapter if JIT compiled successfully
    if (cached_func) {
        std::cout << "[NovaAOT] (Verification) Binary compiled and loaded. Executing with ABI Adapter..." << std::endl;
        task.jit_function = reinterpret_cast<void*>(&JITABIAdapter);
    } else {
        task.jit_function = reinterpret_cast<void*>(&LegacyInterpWrapper);
    }

    // Arguments for Wrapper
    std::vector<void*> exec_inputs;
    exec_inputs.push_back(const_cast<Compiled::Impl*>(p.get()));
    exec_inputs.push_back(const_cast<std::vector<Tensor*>*>(&inputs));
    exec_inputs.push_back(const_cast<std::vector<Tensor*>*>(&params));
    exec_inputs.push_back(cached_func); // args[3] is the real JIT func ptr (nullptr if using legacy)

    task.args = {
        nova::runtime::ArgInput{0},
        nova::runtime::ArgInput{1},
        nova::runtime::ArgInput{2},
        nova::runtime::ArgInput{3}
    };

    plan.tasks.push_back(task);

    // 4. Execute
    // std::cout << "\n[NovaRuntime] Dispatching execution via ExecutionEngine..." << std::endl;
    auto* result_av = engine.Execute(plan, exec_inputs, {});
    result_av->Await();

    // 5. Retrieve Result
    if (result_av->IsError()) {
        std::cerr << "Runtime Integration Error: " << result_av->GetError() << "\n";
        return false;
    }

    auto* concrete = dynamic_cast<nova::runtime::ConcreteAsyncValue<void*>*>(result_av);
    if (!concrete) return false;

    Tensor* result_tensor = static_cast<Tensor*>(concrete->get());
    if (!result_tensor) return false;

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
