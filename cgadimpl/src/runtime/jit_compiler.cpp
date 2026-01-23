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
#include <mutex> // Added for std::mutex
#include "ad/ag_all.hpp"
#include "ad/ops/ops.hpp"
#include "mlp/activation.h"
#include <fstream>
#include <dlfcn.h>
#include <random>
#include <cuda_runtime.h>

namespace ag::jit {

struct ResultMetadata {
    std::vector<int64_t> shape;
    OwnTensor::Dtype dtype;
    OwnTensor::DeviceIndex device;
};

struct AOTContext {
    void* ciface_func = nullptr;
    void* dl_handle = nullptr;
    Plan plan;
    std::string mlir_source;
    std::vector<ResultMetadata> result_meta;

    ~AOTContext() {
        if (dl_handle) {
            dlclose(dl_handle);
        }
    }
};

static size_t getElementSize(OwnTensor::Dtype dtype) {
    switch (dtype) {
        case OwnTensor::Dtype::Float32: return 4;
        case OwnTensor::Dtype::Float16: return 2;
        case OwnTensor::Dtype::Bfloat16: return 2;
        case OwnTensor::Dtype::Int32: return 4;
        case OwnTensor::Dtype::Int64: return 8;
        default: return 4;
    }
}

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
            case Op::Tanh:       return OwnTensor::trig::tanh(*a[0]);
            case Op::GELU:       return OwnTensor::mlp_forward::GeLU(*a[0]);
            case Op::Sign:       { cudaStream_t stream = (cudaStream_t)ag::current_stream(); return OwnTensor::sign(*a[0], stream); }
            
            case Op::MatMul:     return OwnTensor::matmul(*a[0], *a[1]);

            // Reductions need to be updated to the new API
            case Op::Sum: return OwnTensor::reduce_sum(*a[0]);
            case Op::RowSum: return OwnTensor::reduce_sum(*a[0], {1}, true);
            case Op::RowMax: return OwnTensor::reduce_max(*a[0], {1}, true);
            case Op::MeanAll: return OwnTensor::reduce_mean(*a[0]);

            case Op::MSELoss: {
                Tensor diff = *a[0] - *a[1];
                return OwnTensor::reduce_mean(diff * diff);
            }
            case Op::MAELoss: {
                cudaStream_t stream = (cudaStream_t)ag::current_stream();
                return OwnTensor::reduce_mean(OwnTensor::abs(*a[0] - *a[1], stream));
            }
            case Op::BinaryCrossEntropy:
                return OwnTensor::mlp_forward::binary_cross_entropy(*a[0], *a[1]);
            case Op::CategoricalCrossEntropy:
                return OwnTensor::mlp_forward::categorical_cross_entropy(*a[0], *a[1]);
            case Op::CeWithLogits:
            {
                const Tensor& Z = *a[0];
                const Tensor& Y = *a[1];
                Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
                Tensor z_shifted = Z - max_val;
                Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
                Tensor log_sm = z_shifted - log_sum_exp;
                Tensor prod = Y * log_sm;
                Tensor sum_prod = OwnTensor::reduce_sum(prod, {-1}); 
                return OwnTensor::reduce_mean(sum_prod * -1.0f); 
            }
            case Op::KLDivergence:
            {
                const Tensor& Z = *a[0];
                const Tensor& Y = *a[1];
                Tensor log_Y = OwnTensor::log(Y + 1e-9f, ag::current_stream());
                Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
                Tensor z_shifted = Z - max_val;
                Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
                Tensor log_sm_Z = z_shifted - log_sum_exp;
                Tensor kl_div_elementwise = Y * (log_Y - log_sm_Z);
                Tensor sum_kl = OwnTensor::reduce_sum(kl_div_elementwise, {-1});
                return OwnTensor::reduce_mean(sum_kl);
            }
            case Op::SparseCeWithLogits:
            {
                const Tensor& Z = *a[0];
                const Tensor& Y = *a[1];
                Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
                Tensor z_shifted = Z - max_val;
                Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
                Tensor log_sm_Z = z_shifted - log_sum_exp;
                Tensor selected_log_probs = OwnTensor::gather(log_sm_Z, 1, Y);
                return OwnTensor::reduce_mean(selected_log_probs * -1.0f);
            }
            case Op::Leaf: {
                return *a[0];
            }
            default: {
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

        out = slots[plan.out_slots[0]];
        return true;
    }

    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             std::vector<Tensor>& outs) const {
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

        outs.clear();
        outs.reserve(plan.out_slots.size());
        for (int slot : plan.out_slots) {
            outs.push_back(slots[slot]);
        }
        return true;
    }
};

// Forward declarations for AOT compilation helpers
void* compileAndLoad(const std::string& mlir_source, void** out_dl_handle, const std::string& device);
void storeAOTContext(void* func_ptr, void* dl_handle, const Plan& plan, const std::string& mlir_source, std::shared_ptr<void>& out_context);

static std::vector<int64_t> getMLIRShape(const std::vector<int64_t>& dims, Op op) {
    // Nova compiler expects rank-0 for total reductions that result in a single element
    if (dims.size() == 1 && dims[0] == 1) {
        if (op == Op::Sum || op == Op::MeanAll || op == Op::MSELoss || op == Op::MAELoss ||
            op == Op::BinaryCrossEntropy || op == Op::CategoricalCrossEntropy ||
            op == Op::SparseCeWithLogits || op == Op::CeWithLogits || op == Op::KLDivergence) {
            return {};
        }
    }
    return dims;
}

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
        case Op::MSELoss:                return "nova.mse";
        case Op::MAELoss:                return "nova.mae";
        case Op::BinaryCrossEntropy:     return "nova.bce";
        case Op::CategoricalCrossEntropy: return "nova.cce";
        case Op::CeWithLogits:           return "nova.ce_with_logits";
        case Op::KLDivergence:           return "nova.kldivergence";
        case Op::SparseCeWithLogits:     return "nova.sce";

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

    std::string return_vars;
    std::string return_types;
    
    for (size_t i = 0; i < plan.out_slots.size(); ++i) {
        int slot = plan.out_slots[i];
        return_vars += slot_to_var_name.at(slot);
        
        // Find meta for this slot (inefficient but safe)
        TensorMetadata meta;
        for(const auto& s : plan.steps) if(s.out_slot == slot) { meta = s.out_meta; break; }
        
        auto shape = meta.shape;
        // Scalar reduction check (simplified)
        bool is_scalar = false;
        for(const auto& s : plan.steps) {
            if(s.out_slot == slot && 
               (s.op == Op::Sum || s.op == Op::MeanAll || s.op == Op::MSELoss) && 
               shape.size() == 1 && shape[0] == 1) {
                is_scalar = true;
                break;
            }
        }
        if(is_scalar) shape = {};

        return_types += "tensor<" + shapeToMLIR(shape) + dtypeToMLIR(meta.dtype) + ">";
        
        if (i < plan.out_slots.size() - 1) {
            return_vars += ", ";
            return_types += ", ";
        }
    }

    ss << "  return " << return_vars << " : " << return_types << "\n";
    ss << "}\n";
    return ss.str();
}

// Forward declarations for AOT compilation helpers
void* compileAndLoad(const std::string& mlir_source, void** out_dl_handle, const std::string& device);
void storeAOTContext(void* func_ptr, void* dl_handle, const Plan& plan, const std::string& mlir_source, std::shared_ptr<void>& out_context);

// Mutex for serializing JIT compilation to prevent race conditions
static std::mutex jit_compilation_mutex;

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions& opts) {
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

    // --- Topological Sort ---
    auto order = topo_from(output.node.get());
    std::unordered_map<Node*,int> slot_of;
    slot_of.reserve(order.size());

    for (Node* n : order) {
        if (n->op == Op::Leaf) continue;
        Step st;
        st.op = n->op;
        st.out_meta = ag::jit::TensorMetadata(getMLIRShape(n->shape(), n->op), n->value.dtype(), n->value.device());
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
    plan.out_slots.push_back(slot_of.at(output.node.get()));

    // --- Backward Pass Generation ---
    if (opts.include_backward) {
        // 1. Initialize gradients
        // Map from Node* to slot index of its gradient
        std::unordered_map<Node*, int> grad_slot_of;
        
        // Initialize output gradient (dL/dL = 1.0)
        {
            Step st;
            st.op = Op::Leaf; // Literal
            // Create a scalar 1.0 tensor
            auto dt = output.val().dtype();
            auto dev = output.val().device();
            auto opts = OwnTensor::TensorOptions().with_dtype(dt).with_device(dev);
            Tensor one = OwnTensor::Tensor::ones(Shape{{1}}, opts);
            st.args.push_back(ArgLit{one});
            st.out_meta = ag::jit::TensorMetadata(output.shape(), output.val().dtype(), output.val().device());
            st.out_slot = plan.num_slots++;
            grad_slot_of[output.node.get()] = st.out_slot;
            plan.steps.push_back(std::move(st));
        }

        // 2. Reverse topological sort
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Node* n = *it;
            if (grad_slot_of.find(n) == grad_slot_of.end()) continue; // No gradient for this node
            
            int grad_slot = grad_slot_of[n];

            auto accumulate_grad = [&](Node* node, int slot) {
                if (grad_slot_of.count(node)) {
                    Step acc;
                    acc.op = Op::Add;
                    acc.args.push_back(ArgSlot{grad_slot_of[node]});
                    acc.args.push_back(ArgSlot{slot});
                    acc.out_meta = ag::jit::TensorMetadata(node->shape(), node->value.dtype(), node->value.device());
                    acc.out_slot = plan.num_slots++;
                    grad_slot_of[node] = acc.out_slot;
                    plan.steps.push_back(std::move(acc));
                } else {
                    grad_slot_of[node] = slot;
                }
            };
            
            // Generate backward steps based on op
            if (n->op == Op::Add || n->op == Op::Sub) {
                // z = x + y => dx = dz, dy = dz (or -dz for Sub)
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    Node* input = n->inputs[i].get();
                    if (input->requires_grad()) {
                        int current_grad_slot = grad_slot;
                        
                        // Handle Sub second operand
                        if (n->op == Op::Sub && i == 1) {
                            Step neg;
                            neg.op = Op::Leaf;
                            Tensor neg_one = OwnTensor::Tensor::full(Shape{{1}}, ag::options(input->value), -1.0f);
                            neg.args.push_back(ArgLit{neg_one});
                            neg.out_meta = ag::jit::TensorMetadata(std::vector<int64_t>{}, input->value.dtype(), input->value.device());
                            neg.out_slot = plan.num_slots++;
                            plan.steps.push_back(neg);
                            
                            Step st;
                            st.op = Op::Mul;
                            st.args.push_back(ArgSlot{grad_slot});
                            st.args.push_back(ArgSlot{neg.out_slot});
                            st.out_meta = ag::jit::TensorMetadata(n->shape(), input->value.dtype(), input->value.device());
                            st.out_slot = plan.num_slots++;
                            plan.steps.push_back(st);
                            current_grad_slot = st.out_slot;
                        }

                        // Handle broadcasting: if input shape is {1, C} and output is {B, C}, sum over dim 0
                        if (input->shape().size() == 2 && n->shape().size() == 2 && 
                            input->shape()[0] == 1 && n->shape()[0] > 1) {
                            
                            // 1. Transpose dz: {B, C} -> {C, B}
                            Step t1;
                            t1.op = Op::Transpose;
                            t1.args.push_back(ArgSlot{current_grad_slot});
                            t1.out_meta = ag::jit::TensorMetadata( {n->shape()[1], n->shape()[0]}, n->value.dtype(), n->value.device());
                            t1.out_slot = plan.num_slots++;
                            plan.steps.push_back(t1);
                            
                            // 2. RowSum: {C, B} -> {C, 1}
                            Step r1;
                            r1.op = Op::RowSum;
                            r1.args.push_back(ArgSlot{t1.out_slot});
                            r1.out_meta = ag::jit::TensorMetadata(std::vector<int64_t>{n->shape()[1], 1}, n->value.dtype(), n->value.device());
                            r1.out_slot = plan.num_slots++;
                            plan.steps.push_back(r1);
                            
                            // 3. Transpose back: {C, 1} -> {1, C}
                            Step t2;
                            t2.op = Op::Transpose;
                            t2.args.push_back(ArgSlot{r1.out_slot});
                            t2.out_meta = ag::jit::TensorMetadata(std::vector<int64_t>{1, n->shape()[1]}, n->value.dtype(), n->value.device());
                            t2.out_slot = plan.num_slots++;
                            plan.steps.push_back(t2);
                            
                            current_grad_slot = t2.out_slot;
                        }
                        // Handle broadcasting: if input shape is {B, 1} and output is {B, C}, sum over dim 1
                        else if (input->shape().size() == 2 && n->shape().size() == 2 && 
                                 input->shape()[1] == 1 && n->shape()[1] > 1) {
                            // RowSum: {B, C} -> {B, 1}
                            Step r1;
                            r1.op = Op::RowSum;
                            r1.args.push_back(ArgSlot{current_grad_slot});
                            r1.out_meta = ag::jit::TensorMetadata(std::vector<int64_t>{n->shape()[0], 1}, n->value.dtype(), n->value.device());
                            r1.out_slot = plan.num_slots++;
                            plan.steps.push_back(r1);
                            current_grad_slot = r1.out_slot;
                        }

                        accumulate_grad(input, current_grad_slot);
                    }
                }
            } else if (n->op == Op::Mul) {
                // z = x * y => dx = dz * y, dy = dz * x
                for (size_t i = 0; i < 2; ++i) {
                    Node* input = n->inputs[i].get();
                    Node* other = n->inputs[1 - i].get();
                    if (input->requires_grad()) {
                        Step st;
                        st.op = Op::Mul;
                        st.args.push_back(ArgSlot{grad_slot});
                        
                        if (slot_of.count(other)) st.args.push_back(ArgSlot{slot_of[other]});
                        else if (is_in(in_ix, other)) st.args.push_back(ArgInput{in_ix[other]});
                        else if (is_in(par_ix, other)) st.args.push_back(ArgParam{par_ix[other]});
                        else st.args.push_back(ArgLit{other->value});
                        
                        st.out_meta = ag::jit::TensorMetadata(n->shape(), n->value.dtype(), n->value.device());
                        st.out_slot = plan.num_slots++;
                        plan.steps.push_back(st);
                        
                        int current_grad_slot = st.out_slot;
                        
                        // Handle broadcasting (same as Add)
                        if (input->shape().size() == 2 && n->shape().size() == 2 && 
                            input->shape()[0] == 1 && n->shape()[0] > 1) {
                            Step t1; t1.op = Op::Transpose; t1.args.push_back(ArgSlot{current_grad_slot});
                            t1.out_meta = ag::jit::TensorMetadata( {n->shape()[1], n->shape()[0]}, n->value.dtype(), n->value.device());
                            t1.out_slot = plan.num_slots++; plan.steps.push_back(t1);
                            Step r1; r1.op = Op::RowSum; r1.args.push_back(ArgSlot{t1.out_slot});
                            r1.out_meta = ag::jit::TensorMetadata( {n->shape()[1], 1}, n->value.dtype(), n->value.device());
                            r1.out_slot = plan.num_slots++; plan.steps.push_back(r1);
                            Step t2; t2.op = Op::Transpose; t2.args.push_back(ArgSlot{r1.out_slot});
                            t2.out_meta = ag::jit::TensorMetadata( {1, n->shape()[1]}, n->value.dtype(), n->value.device());
                            t2.out_slot = plan.num_slots++; plan.steps.push_back(t2);
                            current_grad_slot = t2.out_slot;
                        } else if (input->shape().size() == 2 && n->shape().size() == 2 && 
                                   input->shape()[1] == 1 && n->shape()[1] > 1) {
                            Step r1; r1.op = Op::RowSum; r1.args.push_back(ArgSlot{current_grad_slot});
                            r1.out_meta = ag::jit::TensorMetadata( {n->shape()[0], 1}, n->value.dtype(), n->value.device());
                            r1.out_slot = plan.num_slots++; plan.steps.push_back(r1);
                            current_grad_slot = r1.out_slot;
                        }
                        
                        accumulate_grad(input, current_grad_slot);
                    }
                }
            } else if (n->op == Op::MatMul) {
                // z = x @ y
                Node* x = n->inputs[0].get();
                Node* y = n->inputs[1].get();
                
                // dx = dz @ y.T
                if (x->requires_grad()) {
                    Step st;
                    st.op = Op::MatMul;
                    st.args.push_back(ArgSlot{grad_slot});
                    
                    Step t_op;
                    t_op.op = Op::Transpose;
                    if (slot_of.count(y)) t_op.args.push_back(ArgSlot{slot_of[y]});
                    else if (is_in(in_ix, y)) t_op.args.push_back(ArgInput{in_ix[y]});
                    else if (is_in(par_ix, y)) t_op.args.push_back(ArgParam{par_ix[y]});
                    else t_op.args.push_back(ArgLit{y->value});
                    
                    t_op.out_meta = ag::jit::TensorMetadata( {y->shape()[1], y->shape()[0]}, y->value.dtype(), y->value.device());
                    t_op.out_slot = plan.num_slots++;
                    plan.steps.push_back(t_op);
                    
                    st.args.push_back(ArgSlot{t_op.out_slot});
                    st.out_meta = ag::jit::TensorMetadata(x->shape(), x->value.dtype(), x->value.device());
                    st.out_slot = plan.num_slots++;
                    
                    if (grad_slot_of.count(x)) {
                        Step acc;
                        acc.op = Op::Add;
                        acc.args.push_back(ArgSlot{grad_slot_of[x]});
                        acc.args.push_back(ArgSlot{st.out_slot});
                        acc.out_meta = st.out_meta;
                        acc.out_slot = plan.num_slots++;
                        plan.steps.push_back(std::move(st));
                        plan.steps.push_back(std::move(acc));
                        grad_slot_of[x] = acc.out_slot;
                    } else {
                        plan.steps.push_back(std::move(st));
                        grad_slot_of[x] = st.out_slot;
                    }
                }
                
                // dy = x.T @ dz
                if (y->requires_grad()) {
                    Step st;
                    st.op = Op::MatMul;
                    
                    Step t_op;
                    t_op.op = Op::Transpose;
                    if (slot_of.count(x)) t_op.args.push_back(ArgSlot{slot_of[x]});
                    else if (is_in(in_ix, x)) t_op.args.push_back(ArgInput{in_ix[x]});
                    else if (is_in(par_ix, x)) t_op.args.push_back(ArgParam{par_ix[x]});
                    else t_op.args.push_back(ArgLit{x->value});
                    
                    t_op.out_meta = ag::jit::TensorMetadata( {x->shape()[1], x->shape()[0]}, x->value.dtype(), x->value.device());
                    t_op.out_slot = plan.num_slots++;
                    plan.steps.push_back(t_op);
                    
                    st.args.push_back(ArgSlot{t_op.out_slot});
                    st.args.push_back(ArgSlot{grad_slot});
                    st.out_meta = ag::jit::TensorMetadata(y->shape(), y->value.dtype(), y->value.device());
                    st.out_slot = plan.num_slots++;
                    plan.steps.push_back(std::move(st));
                    accumulate_grad(y, st.out_slot);
                }
            } else if (n->op == Op::Relu) {
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                    // dx = dz * (x > 0 ? 1 : 0)
                    // We implement this as dz * Sign(Relu(x))
                    
                    // 1. Relu(x)
                    Step relu_step;
                    relu_step.op = Op::Relu;
                    if (slot_of.count(input)) relu_step.args.push_back(ArgSlot{slot_of[input]});
                    else if (is_in(in_ix, input)) relu_step.args.push_back(ArgInput{in_ix[input]});
                    else if (is_in(par_ix, input)) relu_step.args.push_back(ArgParam{par_ix[input]});
                    else relu_step.args.push_back(ArgLit{input->value});
                    relu_step.out_meta = ag::jit::TensorMetadata( input->shape(), input->value.dtype(), input->value.device());
                    relu_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(relu_step);
                    
                    // 2. Sign(Relu(x))
                    Step sign_step;
                    sign_step.op = Op::Sign;
                    sign_step.args.push_back(ArgSlot{relu_step.out_slot});
                    sign_step.out_meta = relu_step.out_meta;
                    sign_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(sign_step);
                    
                    // 3. dz * Sign(Relu(x))
                    Step mul_step;
                    mul_step.op = Op::Mul;
                    mul_step.args.push_back(ArgSlot{grad_slot});
                    mul_step.args.push_back(ArgSlot{sign_step.out_slot});
                    mul_step.out_meta = sign_step.out_meta;
                    mul_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(mul_step);
                    
                    if (grad_slot_of.count(input)) {
                        Step acc;
                        acc.op = Op::Add;
                        acc.args.push_back(ArgSlot{grad_slot_of[input]});
                        acc.args.push_back(ArgSlot{mul_step.out_slot});
                        acc.out_meta = mul_step.out_meta;
                        acc.out_slot = plan.num_slots++;
                        grad_slot_of[input] = acc.out_slot;
                        plan.steps.push_back(std::move(acc));
                    } else {
                        grad_slot_of[input] = mul_step.out_slot;
                    }
                }
            } else if (n->op == Op::GELU) {
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    // For now, let's use a simpler approximation: GELU'(x) = sigmoid(1.702 * x)
                    // This is a common approximation for GELU derivative.
                    
                    // 1. Constant 1.702
                    Step c1;
                    c1.op = Op::Leaf;
                    Tensor c1_t = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(input->value.dtype()).with_device(input->value.device()), 1.702f);
                    c1.args.push_back(ArgLit{c1_t});
                    c1.out_meta = ag::jit::TensorMetadata( {}, input->value.dtype(), input->value.device());
                    c1.out_slot = plan.num_slots++;
                    plan.steps.push_back(c1);
                    
                    // 2. 1.702 * x
                    Step m1;
                    m1.op = Op::Mul;
                    if (slot_of.count(input)) m1.args.push_back(ArgSlot{slot_of[input]});
                    else if (is_in(in_ix, input)) m1.args.push_back(ArgInput{in_ix[input]});
                    else if (is_in(par_ix, input)) m1.args.push_back(ArgParam{par_ix[input]});
                    else m1.args.push_back(ArgLit{input->value});
                    m1.args.push_back(ArgSlot{c1.out_slot});
                    m1.out_meta = ag::jit::TensorMetadata( input->shape(), input->value.dtype(), input->value.device());
                    m1.out_slot = plan.num_slots++;
                    plan.steps.push_back(m1);
                    
                    // 3. sigmoid(1.702 * x) = 0.5 * (1 + tanh(0.5 * 1.702 * x))
                    // We can use Op::Tanh if we implement sigmoid via tanh
                    // Or just use a simpler approximation if Op::Sigmoid existed.
                    // Since we have Op::Tanh: sigmoid(y) = 0.5 * (tanh(y/2) + 1)
                    
                    // 3.1. y/2 = 0.851 * x
                    Step c2;
                    c2.op = Op::Leaf;
                    Tensor c2_t = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(input->value.dtype()).with_device(input->value.device()), 0.851f);
                    c2.args.push_back(ArgLit{c2_t});
                    c2.out_meta = ag::jit::TensorMetadata( {}, input->value.dtype(), input->value.device());
                    c2.out_slot = plan.num_slots++;
                    plan.steps.push_back(c2);
                    
                    Step m2;
                    m2.op = Op::Mul;
                    m2.args.push_back(ArgSlot{m1.out_slot}); // This is 1.702*x, wait I need 0.851*x
                    // Actually m1 is 1.702*x. m1/2 is 0.851*x.
                    // Let's just use input * 0.851
                    m2.args.clear();
                    if (slot_of.count(input)) m2.args.push_back(ArgSlot{slot_of[input]});
                    else if (is_in(in_ix, input)) m2.args.push_back(ArgInput{in_ix[input]});
                    else if (is_in(par_ix, input)) m2.args.push_back(ArgParam{par_ix[input]});
                    else m2.args.push_back(ArgLit{input->value});
                    m2.args.push_back(ArgSlot{c2.out_slot});
                    m2.out_meta = m1.out_meta;
                    m2.out_slot = plan.num_slots++;
                    plan.steps.push_back(m2);
                    
                    // 3.2. tanh(0.851 * x)
                    Step t1;
                    t1.op = Op::Tanh;
                    t1.args.push_back(ArgSlot{m2.out_slot});
                    t1.out_meta = m2.out_meta;
                    t1.out_slot = plan.num_slots++;
                    plan.steps.push_back(t1);
                    
                    // 3.3. tanh + 1
                    Step c3;
                    c3.op = Op::Leaf;
                    Tensor c3_t = OwnTensor::Tensor::ones(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(input->value.dtype()).with_device(input->value.device()));
                    c3.args.push_back(ArgLit{c3_t});
                    c3.out_meta = ag::jit::TensorMetadata( {}, input->value.dtype(), input->value.device());
                    c3.out_slot = plan.num_slots++;
                    plan.steps.push_back(c3);
                    
                    Step a1;
                    a1.op = Op::Add;
                    a1.args.push_back(ArgSlot{t1.out_slot});
                    a1.args.push_back(ArgSlot{c3.out_slot});
                    a1.out_meta = t1.out_meta;
                    a1.out_slot = plan.num_slots++;
                    plan.steps.push_back(a1);
                    
                    // 3.4. 0.5 * (tanh + 1)
                    Step c4;
                    c4.op = Op::Leaf;
                    Tensor c4_t = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(input->value.dtype()).with_device(input->value.device()), 0.5f);
                    c4.args.push_back(ArgLit{c4_t});
                    c4.out_meta = ag::jit::TensorMetadata( {}, input->value.dtype(), input->value.device());
                    c4.out_slot = plan.num_slots++;
                    plan.steps.push_back(c4);
                    
                    Step m3;
                    m3.op = Op::Mul;
                    m3.args.push_back(ArgSlot{a1.out_slot});
                    m3.args.push_back(ArgSlot{c4.out_slot});
                    m3.out_meta = a1.out_meta;
                    m3.out_slot = plan.num_slots++;
                    plan.steps.push_back(m3);
                    
                    // 4. grad * sigmoid(1.702 * x)
                    Step final_grad;
                    final_grad.op = Op::Mul;
                    final_grad.args.push_back(ArgSlot{grad_slot});
                    final_grad.args.push_back(ArgSlot{m3.out_slot});
                    final_grad.out_meta = m3.out_meta;
                    final_grad.out_slot = plan.num_slots++;
                    plan.steps.push_back(final_grad);
                    
                    if (grad_slot_of.count(input)) {
                        Step acc;
                        acc.op = Op::Add;
                        acc.args.push_back(ArgSlot{grad_slot_of[input]});
                        acc.args.push_back(ArgSlot{final_grad.out_slot});
                        acc.out_meta = final_grad.out_meta;
                        acc.out_slot = plan.num_slots++;
                        grad_slot_of[input] = acc.out_slot;
                        plan.steps.push_back(std::move(acc));
                    } else {
                        grad_slot_of[input] = final_grad.out_slot;
                    }
                }
            } else if (n->op == Op::MSELoss) {
                Node* x = n->inputs[0].get();
                Node* y = n->inputs[1].get();
                int64_t N = 1;
                for (auto d : x->shape()) N *= d;
                
                Step const_step;
                const_step.op = Op::Leaf;
                float scale = 2.0f / N;
                Tensor scale_t = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(x->value.dtype()).with_device(x->value.device()), scale);
                const_step.args.push_back(ArgLit{scale_t});
                const_step.out_meta = ag::jit::TensorMetadata( {}, x->value.dtype(), x->value.device());
                const_step.out_slot = plan.num_slots++;
                plan.steps.push_back(const_step);
                int scale_slot = const_step.out_slot;

                Step sub_step;
                sub_step.op = Op::Sub;
                if (slot_of.count(x)) sub_step.args.push_back(ArgSlot{slot_of[x]});
                else if (is_in(in_ix, x)) sub_step.args.push_back(ArgInput{in_ix[x]});
                else if (is_in(par_ix, x)) sub_step.args.push_back(ArgParam{par_ix[x]});
                else sub_step.args.push_back(ArgLit{x->value});

                if (slot_of.count(y)) sub_step.args.push_back(ArgSlot{slot_of[y]});
                else if (is_in(in_ix, y)) sub_step.args.push_back(ArgInput{in_ix[y]});
                else if (is_in(par_ix, y)) sub_step.args.push_back(ArgParam{par_ix[y]});
                else sub_step.args.push_back(ArgLit{y->value});
                
                sub_step.out_meta = ag::jit::TensorMetadata(x->shape(), x->value.dtype(), x->value.device());
                sub_step.out_slot = plan.num_slots++;
                plan.steps.push_back(sub_step);
                int diff_slot = sub_step.out_slot;

                Step mul_step;
                mul_step.op = Op::Mul;
                mul_step.args.push_back(ArgSlot{diff_slot});
                mul_step.args.push_back(ArgSlot{scale_slot});
                mul_step.out_meta = sub_step.out_meta;
                mul_step.out_slot = plan.num_slots++;
                plan.steps.push_back(mul_step);
                int grad_common = mul_step.out_slot;
                
                if (x->requires_grad()) {
                    Step st;
                    st.op = Op::Mul;
                    st.args.push_back(ArgSlot{grad_common});
                    st.args.push_back(ArgSlot{grad_slot});
                    st.out_meta = ag::jit::TensorMetadata(x->shape(), x->value.dtype(), x->value.device());
                    st.out_slot = plan.num_slots++;
                    plan.steps.push_back(std::move(st));
                    accumulate_grad(x, st.out_slot);
                }
                
                if (y->requires_grad()) {
                    Step neg;
                    neg.op = Op::Leaf; 
                     Tensor neg_one = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(y->value.dtype()).with_device(y->value.device()), -1.0f);
                    neg.args.push_back(ArgLit{neg_one});
                    neg.out_meta = ag::jit::TensorMetadata(std::vector<int64_t>{}, y->value.dtype(), y->value.device());
                    neg.out_slot = plan.num_slots++;
                    plan.steps.push_back(neg);
                    
                    Step st;
                    st.op = Op::Mul;
                    st.args.push_back(ArgSlot{grad_common});
                    st.args.push_back(ArgSlot{neg.out_slot});
                    st.out_meta = ag::jit::TensorMetadata(y->shape(), y->value.dtype(), y->value.device());
                    st.out_slot = plan.num_slots++;
                    plan.steps.push_back(st);
                    int dy_raw = st.out_slot;

                    Step final_dy;
                    final_dy.op = Op::Mul;
                    final_dy.args.push_back(ArgSlot{dy_raw});
                    final_dy.args.push_back(ArgSlot{grad_slot});
                    final_dy.out_meta = ag::jit::TensorMetadata(y->shape(), y->value.dtype(), y->value.device());
                    final_dy.out_slot = plan.num_slots++;
                    plan.steps.push_back(std::move(final_dy));
                    accumulate_grad(y, final_dy.out_slot);
                }
            } else if (n->op == Op::Sum) {
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                    // dx = dz (broadcasted)
                    Step ones_step;
                    ones_step.op = Op::Leaf;
                    Tensor ones = OwnTensor::Tensor::ones(Shape{input->shape()}, ag::options(input->value));
                    ones_step.args.push_back(ArgLit{ones});
                    ones_step.out_meta = ag::jit::TensorMetadata( input->shape(), input->value.dtype(), input->value.device());
                    ones_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(ones_step);
                    
                    Step mul_step;
                    mul_step.op = Op::Mul;
                    mul_step.args.push_back(ArgSlot{grad_slot});
                    mul_step.args.push_back(ArgSlot{ones_step.out_slot});
                    mul_step.out_meta = ones_step.out_meta;
                    mul_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(mul_step);
                    
                    accumulate_grad(input, mul_step.out_slot);
                }
            } else if (n->op == Op::MeanAll) {
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                    int64_t N = 1;
                    for (auto d : input->shape()) N *= d;
                    
                    Step scale_step;
                    scale_step.op = Op::Leaf;
                    Tensor scale = OwnTensor::Tensor::full(Shape{{1}}, ag::options(input->value), 1.0f / N);
                    scale_step.args.push_back(ArgLit{scale});
                    scale_step.out_meta = ag::jit::TensorMetadata( {1}, input->value.dtype(), input->value.device());
                    scale_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(scale_step);
                    
                    Step dz_scaled;
                    dz_scaled.op = Op::Mul;
                    dz_scaled.args.push_back(ArgSlot{grad_slot});
                    dz_scaled.args.push_back(ArgSlot{scale_step.out_slot});
                    dz_scaled.out_meta = ag::jit::TensorMetadata( {1}, input->value.dtype(), input->value.device());
                    dz_scaled.out_slot = plan.num_slots++;
                    plan.steps.push_back(dz_scaled);
                    
                    Step ones_step;
                    ones_step.op = Op::Leaf;
                    Tensor ones = OwnTensor::Tensor::ones(Shape{input->shape()}, ag::options(input->value));
                    ones_step.args.push_back(ArgLit{ones});
                    ones_step.out_meta = ag::jit::TensorMetadata( input->shape(), input->value.dtype(), input->value.device());
                    ones_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(ones_step);
                    
                    Step mul_step;
                    mul_step.op = Op::Mul;
                    mul_step.args.push_back(ArgSlot{dz_scaled.out_slot});
                    mul_step.args.push_back(ArgSlot{ones_step.out_slot});
                    mul_step.out_meta = ones_step.out_meta;
                    mul_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(mul_step);
                    
                    accumulate_grad(input, mul_step.out_slot);
                }
            } else if (n->op == Op::RowSum) {
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                    // dx = dz (broadcasted along columns)
                    Step ones_step;
                    ones_step.op = Op::Leaf;
                    Tensor ones = OwnTensor::Tensor::ones(Shape{input->shape()}, ag::options(input->value));
                    ones_step.args.push_back(ArgLit{ones});
                    ones_step.out_meta = ag::jit::TensorMetadata(input->shape(), input->value.dtype(), input->value.device());
                    ones_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(ones_step);
                    
                    Step mul_step;
                    mul_step.op = Op::Mul;
                    mul_step.args.push_back(ArgSlot{grad_slot});
                    mul_step.args.push_back(ArgSlot{ones_step.out_slot});
                    mul_step.out_meta = ones_step.out_meta;
                    mul_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(mul_step);
                    
                    accumulate_grad(input, mul_step.out_slot);
                }
            } else if (n->op == Op::Transpose) {
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                    // dx = dz.transpose() (assuming 2D for now)
                    Step t_step;
                    t_step.op = Op::Transpose;
                    t_step.args.push_back(ArgSlot{grad_slot});
                    t_step.out_meta = ag::jit::TensorMetadata(input->shape(), input->value.dtype(), input->value.device());
                    t_step.out_slot = plan.num_slots++;
                    plan.steps.push_back(t_step);
                    
                    accumulate_grad(input, t_step.out_slot);
                }
            }
        }
        
        // 3. Collect parameter gradients
        for (const auto& param : params) {
            Node* p = param.node.get();
            if (grad_slot_of.count(p)) {
                plan.out_slots.push_back(grad_slot_of[p]);
            } else {
                Step st;
                st.op = Op::Leaf;
                Tensor z = OwnTensor::Tensor::zeros(p->value.shape(), ag::options(p->value));
                st.args.push_back(ArgLit{z});
                st.out_meta = ag::jit::TensorMetadata(p->shape(), p->value.dtype(), p->value.device());
                st.out_slot = plan.num_slots++;
                plan.steps.push_back(st);
                plan.out_slots.push_back(st.out_slot);
            }
        }
    }

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
    // c.mlir_source = std::move(generated_mlir_string);
     c.mlir_source = generated_mlir_opbuilder.empty() ? generated_mlir_string : generated_mlir_opbuilder;
    
    // Detect if any input/param is on GPU
    bool use_gpu = false;
    for (const auto& meta : c.p->plan.sig.in_meta) {
        if (meta.device.device == OwnTensor::Device::CUDA) {
            use_gpu = true;
            break;
        }
    }
    if (!use_gpu) {
        for (const auto& meta : c.p->plan.sig.param_meta) {
            if (meta.device.device == OwnTensor::Device::CUDA) {
                use_gpu = true;
                break;
            }
        }
    }



    if (in_memory_module) {
        try {
            mlir::nova::NovaCompilerAPI compiler;
            mlir::nova::CompilerOptions options;
            options.runFullPipeline = true;
            options.device = use_gpu ? "gpu" : "cpu";
            auto compileResult = compiler.compileString(generated_mlir_opbuilder, "", options);
            if (compileResult.success) {
                generated_mlir_opbuilder = compileResult.output;
                std::cout << "=== Optimized MLIR Generated via NovaCompilerAPI ===\n";
                std::ofstream ofs("optimized.mlir");
                ofs << generated_mlir_opbuilder;
                ofs.close();
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
        void* dl_handle = nullptr;
        void* func_ptr = compileAndLoad(c.mlir_source, &dl_handle, use_gpu ? "gpu" : "cpu");
        
        if (func_ptr) {
            c.compiled_func = func_ptr;
            storeAOTContext(func_ptr, dl_handle, c.p->plan, c.mlir_source, c.aot_context);
        }
    }

    return c;
}

// ===================================================================
// AOT Adapter Function
// ===================================================================


// Generic memref descriptor builder
// Returns a vector of bytes representing the memref descriptor
static size_t alignTo(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

std::vector<int64_t> parseTensorSpec(const std::string& tensor_spec) {
    std::vector<int64_t> shape;
    if (tensor_spec.find('x') == std::string::npos) return shape;
    size_t pos = 0;
    while (pos < tensor_spec.size()) {
        size_t x_pos = tensor_spec.find('x', pos);
        if (x_pos == std::string::npos) break;
        try { shape.push_back(std::stoll(tensor_spec.substr(pos, x_pos - pos))); } catch (...) { break; }
        pos = x_pos + 1;
    }
    return shape;
}

std::vector<std::vector<int64_t>> parseMLIRResultShapes(const std::string& mlir_source) {
    std::vector<std::vector<int64_t>> result_shapes;
    size_t arrow_pos = mlir_source.find("-> ");
    if (arrow_pos == std::string::npos) return result_shapes;
    size_t start = arrow_pos + 3;
    if (mlir_source[start] == '(') {
        size_t close_paren = mlir_source.find(')', start);
        if (close_paren == std::string::npos) return result_shapes;
        std::string tuple_content = mlir_source.substr(start + 1, close_paren - start - 1);
        size_t pos = 0;
        while (pos < tuple_content.size()) {
            while (pos < tuple_content.size() && (tuple_content[pos] == ' ' || tuple_content[pos] == ',')) pos++;
            size_t t_start = tuple_content.find("tensor<", pos);
            if (t_start == std::string::npos) break;
            size_t spec_end = tuple_content.find('>', t_start);
            if (spec_end == std::string::npos) break;
            result_shapes.push_back(parseTensorSpec(tuple_content.substr(t_start + 7, spec_end - t_start - 7)));
            pos = spec_end + 1;
        }
    } else if (mlir_source.substr(start, 7) == "tensor<") {
        size_t spec_end = mlir_source.find('>', start);
        if (spec_end != std::string::npos) result_shapes.push_back(parseTensorSpec(mlir_source.substr(start + 7, spec_end - start - 7)));
    }
    return result_shapes;
}

std::vector<char> buildMemRefDescriptor(Tensor* tensor, const TensorMetadata& meta) {
    size_t rank = meta.shape.size();
    size_t descriptor_size = sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * rank);
    std::vector<char> descriptor(descriptor_size, 0);
    char* ptr = descriptor.data();
    
    // Use raw data pointer (void*)
    void* raw_ptr = tensor->data();
    
    *reinterpret_cast<void**>(ptr) = raw_ptr;
    ptr += sizeof(void*);
    *reinterpret_cast<void**>(ptr) = raw_ptr;
    ptr += sizeof(void*);
    *reinterpret_cast<int64_t*>(ptr) = 0;
    ptr += sizeof(int64_t);
    for (size_t i = 0; i < rank; ++i) {
        *reinterpret_cast<int64_t*>(ptr) = tensor->shape().dims[i];
        ptr += sizeof(int64_t);
    }
    for (size_t i = 0; i < rank; ++i) {
        *reinterpret_cast<int64_t*>(ptr) = tensor->stride().strides[i];
        ptr += sizeof(int64_t);
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

    return result_shape;
}

// FIX: Memory Pool to prevent allocator corruption/race conditions
struct ThreadLocalPool {
    // Pool for packed output argument structures (raw pointers)
    std::vector<std::pair<void*, size_t>> packed_buffers;
    // Pool for result buffers (host only, CUDA managed separately)
    std::vector<std::pair<void*, size_t>> result_buffers;
    
    ~ThreadLocalPool() {
        for (auto& p : packed_buffers) std::free(p.first);
        for (auto& p : result_buffers) std::free(p.first);
    }

    void* getPackedBuffer(size_t size) {
        // Simple best-fit or first-fit? Let's just grab one big enough or alloc new.
        for (auto it = packed_buffers.begin(); it != packed_buffers.end(); ++it) {
            if (it->second >= size) {
                void* ptr = it->first;
                packed_buffers.erase(it);
                return ptr;
            }
        }
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0) return nullptr;
        return ptr;
    }

    void returnPackedBuffer(void* ptr, size_t size) {
        if (ptr) packed_buffers.push_back({ptr, size});
    }

    void* getResultBuffer(size_t size) {
        for (auto it = result_buffers.begin(); it != result_buffers.end(); ++it) {
            if (it->second >= size) {
                void* ptr = it->first;
                result_buffers.erase(it);
                return ptr;
            }
        }
        return std::malloc(size);
    }
    
    void returnResultBuffer(void* ptr, size_t size) {
        if (ptr) result_buffers.push_back({ptr, size});
    }
};

extern "C" void* ABIAdapter(void** args, void* context_ptr) {
    auto* context = static_cast<AOTContext*>(context_ptr);
    if (!context || !context->ciface_func) return nullptr;
    
    // Thread-local pool instance
    thread_local ThreadLocalPool pool;
    
    const Plan& plan = context->plan;
    size_t num_inputs = plan.sig.in_meta.size();
    size_t num_params = plan.sig.param_meta.size();
    
    std::vector<std::vector<char>> input_descriptors;
    for (size_t i = 0; i < num_inputs; ++i) {
        Tensor* t = static_cast<Tensor*>(args[i]);
        input_descriptors.push_back(buildMemRefDescriptor(t, plan.sig.in_meta[i]));
    }
    for (size_t i = 0; i < num_params; ++i) {
        Tensor* t = static_cast<Tensor*>(args[num_inputs + i]);
        input_descriptors.push_back(buildMemRefDescriptor(t, plan.sig.param_meta[i]));
    }
    
    size_t num_results = context->result_meta.size();
    std::vector<size_t> result_desc_offsets;
    size_t packed_struct_size = 0;
    for (size_t r = 0; r < num_results; ++r) {
        packed_struct_size = alignTo(packed_struct_size, 8);
        result_desc_offsets.push_back(packed_struct_size);
        packed_struct_size += sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * context->result_meta[r].shape.size());
    }
    packed_struct_size = alignTo(packed_struct_size, 64);

    packed_struct_size = alignTo(packed_struct_size, 64);

    void* packed_output_raw = pool.getPackedBuffer(packed_struct_size);
    if (!packed_output_raw) {
        return nullptr;
    }
    std::memset(packed_output_raw, 0, packed_struct_size);
    
    std::vector<void*> result_buffers(num_results, nullptr);
    auto cleanup = [&]() {
        for (size_t i = 0; i < num_results; ++i) {
            if (result_buffers[i]) {
                if (context->result_meta[i].device.device == OwnTensor::Device::CUDA) {
                     cudaFree(result_buffers[i]);
                } else {
                     // Get size for returning to pool
                     auto& meta = context->result_meta[i];
                     size_t elem_size = getElementSize(meta.dtype);
                     size_t total_elements = 1;
                     for (auto dim : meta.shape) total_elements *= dim;
                     size_t buffer_size = std::max((size_t)1, total_elements) * elem_size;
                     pool.returnResultBuffer(result_buffers[i], buffer_size);
                }
            }
        }
        if (packed_output_raw) pool.returnPackedBuffer(packed_output_raw, packed_struct_size);
    };

    for (size_t r = 0; r < num_results; ++r) {
        auto& meta = context->result_meta[r];
        size_t elem_size = getElementSize(meta.dtype);
        size_t total_elements = 1;
        for (auto dim : meta.shape) total_elements *= dim;
        size_t buffer_size = std::max((size_t)1, total_elements) * elem_size;
        
        if (meta.device.device == OwnTensor::Device::CUDA) {
            if (cudaMalloc(&result_buffers[r], buffer_size) != cudaSuccess) { cleanup(); return nullptr; }
        } else {
            result_buffers[r] = pool.getResultBuffer(buffer_size);
        }
        if (!result_buffers[r]) { cleanup(); return nullptr; }
        if (meta.device.device != OwnTensor::Device::CUDA) {
            std::memset(result_buffers[r], 0, buffer_size);
        } else {
            cudaMemset(result_buffers[r], 0, buffer_size);
        }
        
        char* desc_ptr = static_cast<char*>(packed_output_raw) + result_desc_offsets[r];
        *reinterpret_cast<void**>(desc_ptr) = result_buffers[r];
        *reinterpret_cast<void**>(desc_ptr + sizeof(void*)) = result_buffers[r];
        *reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2) = 0; // Offset = 0
        
        if (!meta.shape.empty()) {
            int64_t* sizes_ptr = reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2 + sizeof(int64_t));
            int64_t* strides_ptr = sizes_ptr + meta.shape.size();
            for (size_t i = 0; i < meta.shape.size(); ++i) sizes_ptr[i] = meta.shape[i];
            int64_t stride = 1;
            for (int i = (int)meta.shape.size() - 1; i >= 0; --i) { 
                strides_ptr[i] = stride; 
                stride *= meta.shape[i]; 
            }
        }
    }
    
    std::vector<void*> desc_ptrs;
    for (auto& desc : input_descriptors) {
        desc_ptrs.push_back(desc.data());
    }

    using CIfaceFunc = void (*)(void*, void**);
    if (!context->ciface_func) {
        cleanup(); return nullptr;
    }

    reinterpret_cast<CIfaceFunc>(context->ciface_func)(packed_output_raw, desc_ptrs.data());

    auto* output_tensors = new std::vector<Tensor>();
    for (size_t r = 0; r < num_results; ++r) {
        auto& meta = context->result_meta[r];
        char* desc_ptr = static_cast<char*>(packed_output_raw) + result_desc_offsets[r];
        void* allocated = *reinterpret_cast<void**>(desc_ptr);
        void* aligned = *reinterpret_cast<void**>(desc_ptr + sizeof(void*));
        int64_t offset = *reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2);
        
        if (!aligned) {
            continue;
        }

        size_t rank = meta.shape.size();
        int64_t* sizes_ptr = reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2 + sizeof(int64_t));
        int64_t* strides_ptr = sizes_ptr + rank;
        
        Tensor out(OwnTensor::Shape{meta.shape.empty() ? std::vector<int64_t>{1} : meta.shape}, meta.dtype, meta.device, false);
        size_t total_elements = 1; for (auto dim : meta.shape) total_elements *= dim;
        size_t elem_size = getElementSize(meta.dtype);
        
        bool is_contiguous = true;
        int64_t expected_stride = 1;
        for (int i = (int)rank - 1; i >= 0; --i) {
            if (strides_ptr[i] != expected_stride) { is_contiguous = false; break; }
            expected_stride *= (sizes_ptr[i] > 0 ? sizes_ptr[i] : 1);
        }

        if (is_contiguous || rank == 0) {
            cudaMemcpy(out.data(), static_cast<char*>(aligned) + (offset * elem_size), total_elements * elem_size, cudaMemcpyDefault);
        } else {
            if (rank == 1) {
                int64_t s0 = strides_ptr[0];
                char* src_base = static_cast<char*>(aligned) + (offset * elem_size);
                char* dst_base = static_cast<char*>(out.data());
                for (int64_t i = 0; i < meta.shape[0]; ++i) {
                    cudaMemcpy(dst_base + i * elem_size, src_base + i * s0 * elem_size, elem_size, cudaMemcpyDefault);
                }
            } else if (rank == 2) {
                int64_t s0 = strides_ptr[0];
                int64_t s1 = strides_ptr[1];
                char* src_base = static_cast<char*>(aligned) + (offset * elem_size);
                char* dst_base = static_cast<char*>(out.data());
                for (int64_t i = 0; i < meta.shape[0]; ++i) {
                    for (int64_t j = 0; j < meta.shape[1]; ++j) {
                        cudaMemcpy(dst_base + (i * meta.shape[1] + j) * elem_size, 
                                    src_base + (i * s0 + j * s1) * elem_size, elem_size, cudaMemcpyDefault);
                    }
                }
            } else {
                cudaMemcpy(out.data(), static_cast<char*>(aligned) + (offset * elem_size), total_elements * elem_size, cudaMemcpyDefault);
            }
        }
        
        output_tensors->push_back(std::move(out));
        
        if (allocated) {
            cudaPointerAttributes free_attrs;
            cudaError_t free_err = cudaPointerGetAttributes(&free_attrs, allocated);
            bool is_device = (free_err == cudaSuccess && free_attrs.type == cudaMemoryTypeDevice);
            cudaGetLastError(); // Clear error
            
            if (is_device) {
                cudaFree(allocated);
            } else {
                // If the allocated pointer matches our result buffer, return it to pool
                if (allocated == result_buffers[r]) {
                     auto& meta = context->result_meta[r];
                     size_t elem_size = getElementSize(meta.dtype);
                     size_t total_elements = 1;
                     for (auto dim : meta.shape) total_elements *= dim;
                     size_t buffer_size = std::max((size_t)1, total_elements) * elem_size;
                     pool.returnResultBuffer(allocated, buffer_size);
                } else {
                    // Fallback for unknown pointers? Should not happen in this logic.
                    std::free(allocated); 
                }
            }

            if (allocated == result_buffers[r]) {
                result_buffers[r] = nullptr;
            }
        }
    }
    
    cleanup();
    return output_tensors;
}



// Wrapper Declaration (must be global)
extern "C" void* LegacyInterpWrapper(void**);

void* compileAndLoad(const std::string& mlir_source, void** out_dl_handle, const std::string& device) {
    // FIX: Use thread_local for RNG to avoid race conditions in multi-threaded environment
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    
    // FIX: Serialize compilation to prevent file collisions and system call races
    std::lock_guard<std::mutex> lock(jit_compilation_mutex);
    
    std::stringstream ss;
    ss << "/tmp/nova_jit_" << std::hex << dist(gen);
    std::string base_path = ss.str();
    
    std::string mlir_file = base_path + ".mlir";
    std::string ll_file = base_path + ".ll"; // NEW: for LLVM IR
    std::string obj_file = base_path + ".o";
    std::string so_file = base_path + ".so";
    
    // Detect if the input is already LLVM IR (assembly) vs MLIR
    bool is_llvm_ir = (mlir_source.find("target triple") != std::string::npos && 
                        mlir_source.find("module attributes") == std::string::npos &&
                        mlir_source.find("module {") == std::string::npos);

    {
        std::ofstream out(is_llvm_ir ? ll_file : mlir_file);
        out << mlir_source;
        out.close();
    }
    
    if (!is_llvm_ir) {
        #ifndef NOVA_OPT_BIN
            #define NOVA_OPT_BIN "nova-opt"
        #endif
        std::string nova_opt = NOVA_OPT_BIN;
        bool success = mlir::nova::NovaCompilerSystemAPI::compileToObject(mlir_file, obj_file, nova_opt, device);
        if (!success) {
            std::remove(mlir_file.c_str());
            return nullptr;
        }
    } else {
        // Input is already LLVM IR, compile directly to object file
        std::string llc_cmd = "llc " + ll_file + " -relocation-model=pic -filetype=obj -o " + obj_file;
        if (system(llc_cmd.c_str()) != 0) {
            std::cerr << "Error: LLC failed for LLVM IR: " << llc_cmd << "\n";
            std::remove(ll_file.c_str());
            return nullptr;
        }
    }
    
    std::string link_flags;
    if (device == "gpu") {
        const char* llvm_env = std::getenv("LLVM_DIR");
        std::string llvm_lib_dir = llvm_env ? std::string(llvm_env) + "/lib" : "/home/blu-bridge006/Desktop/llvm-project/build/lib";
        link_flags = " -L" + llvm_lib_dir + " -lmlir_cuda_runtime -lmlir_runner_utils -lmlir_c_runner_utils" +
                     " -L/usr/local/cuda/lib64 -lcudart" +
                     " -Wl,-rpath," + llvm_lib_dir +
                     " -Wl,-rpath,/usr/local/cuda/lib64";
    }
    std::string link_cmd = "g++ -shared -fPIC -o " + so_file + " " + obj_file + link_flags + " -ldl -lm";

    if (system(link_cmd.c_str()) != 0) {
        std::cerr << "Error: Linking JIT shared library failed: " << link_cmd << "\n";
        std::remove(mlir_file.c_str());
        std::remove(ll_file.c_str());
        std::remove(obj_file.c_str());
        return nullptr;
    }
    
    void* handle = dlopen(so_file.c_str(), RTLD_NOW | RTLD_GLOBAL);
    
    // Clean up temporary compilation artifacts
    std::remove(mlir_file.c_str());
    std::remove(ll_file.c_str());
    std::remove(obj_file.c_str());
    std::remove(so_file.c_str()); 

    if (!handle) {
        std::cerr << "Error: dlopen failed: " << dlerror() << "\n";
        return nullptr;
    }
    
    if (out_dl_handle) *out_dl_handle = handle;
    
    // Search for entry point symbol (mangled or unmangled)
    const char* symbols[] = {"_mlir_ciface_main", "mlir_ciface_main", "main"};
    void* func_ptr = nullptr;
    for (const char* sym : symbols) {
        if ((func_ptr = dlsym(handle, sym))) break;
    }

    if (!func_ptr) {
        std::cerr << "Error: Could not find any executable symbol in JIT library\n";
        dlclose(handle);
        if (out_dl_handle) *out_dl_handle = nullptr;
        return nullptr;
    }
    
    return func_ptr;
}

void storeAOTContext(void* func_ptr, void* dl_handle, const Plan& plan, const std::string& mlir_source, std::shared_ptr<void>& out_context) {
    auto* ctx = new AOTContext();
    ctx->ciface_func = func_ptr;
    ctx->dl_handle = dl_handle;
    ctx->plan = plan;
    ctx->mlir_source = mlir_source;

    // Cache result metadata to avoid re-calculating on every run
    for (int slot : plan.out_slots) {
        bool found = false;
        for (const auto& step : plan.steps) {
            if (step.out_slot == slot) {
                ctx->result_meta.push_back({step.out_meta.shape, step.out_meta.dtype, step.out_meta.device});
                found = true;
                break;
            }
        }
        if (!found) {
            ctx->result_meta.push_back({{}, OwnTensor::Dtype::Float32, OwnTensor::DeviceIndex(OwnTensor::Device::CPU)});
        }
    }

    out_context = std::shared_ptr<AOTContext>(ctx); 
}

// Wrapper to make the legacy 'run' look like a JIT compiled function
extern "C" void* LegacyInterpWrapper(void** args) {
    auto* impl = static_cast<Compiled::Impl*>(args[0]);
    auto* inputs = static_cast<const std::vector<Tensor*>*>(args[1]);
    auto* params = static_cast<const std::vector<Tensor*>*>(args[2]);
    
    Tensor* out = new Tensor();
    bool success = impl->run(*inputs, *params, *out);
    if (!success) {
        delete out;
        return nullptr;
    }
    return out;
}

bool Compiled::run(const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& params,
                   Tensor& out) const {
    return p->run(inputs, params, out);
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

bool Compiled::run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& params, std::vector<Tensor>& outs) const {
    if (!compiled_func) {
        std::cerr << "Error: JIT function was not compiled.\n";
        return false;
    }
    
    // Ensure all inputs and params are contiguous for JIT execution
    std::vector<Tensor> contiguous_inputs;
    std::vector<Tensor*> final_inputs;
    for (auto* t : inputs) {
        if (!t->is_contiguous()) {
            contiguous_inputs.push_back(t->contiguous());
            final_inputs.push_back(&contiguous_inputs.back());
        } else {
            final_inputs.push_back(t);
        }
    }
    
    std::vector<Tensor> contiguous_params;
    std::vector<Tensor*> final_params;
    for (auto* t : params) {
        if (!t->is_contiguous()) {
            contiguous_params.push_back(t->contiguous());
            final_params.push_back(&contiguous_params.back());
        } else {
            final_params.push_back(t);
        }
    }

    std::vector<void*> exec_inputs; 
    exec_inputs.reserve(final_inputs.size() + final_params.size());
    for (auto* t : final_inputs) exec_inputs.push_back(t); 
    for (auto* t : final_params) exec_inputs.push_back(t);
    
    auto* result_tensors = static_cast<std::vector<Tensor>*>(ABIAdapter(exec_inputs.data(), aot_context.get()));
    if (!result_tensors) return false;
    
    for (auto& tensor : *result_tensors) {
        // If the result is a scalar reduction, move it to CPU to match test expectations.
        if (tensor.is_cuda() && tensor.shape().dims.empty()) {
            outs.push_back(tensor.to_cpu());
        } else {
            outs.push_back(std::move(tensor));
        }
    }
    delete result_tensors; 
    return true;
}

} // namespace ag::jit