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
#include "ad/ag_all.hpp"
#include "ad/ops/ops.hpp"
#include "mlp/activation.h"
#include <fstream>
#include <dlfcn.h>

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
            case Op::Tanh:       return OwnTensor::trig::tanh(*a[0]);
            case Op::GELU:       return OwnTensor::mlp_forward::GeLU(*a[0]);
            
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
void* compileAndLoad(const std::string& mlir_source);
void storeAOTContext(void* func_ptr, const Plan& plan, const std::string& mlir_source);

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
            st.out_meta = {output.shape(), output.val().dtype(), output.val().device()};
            st.out_slot = plan.num_slots++;
            grad_slot_of[output.node.get()] = st.out_slot;
            plan.steps.push_back(std::move(st));
        }

        // 2. Reverse topological sort
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Node* n = *it;
            if (grad_slot_of.find(n) == grad_slot_of.end()) continue; // No gradient for this node
            
            int grad_slot = grad_slot_of[n];
            
            // Generate backward steps based on op
            if (n->op == Op::Add) {
                // z = x + y => dx = dz, dy = dz
                for (auto& input : n->inputs) {
                    if (input->requires_grad()) {
                        int current_grad_slot = grad_slot;
                        
                        // Handle broadcasting: if input shape is {1, C} and output is {B, C}, sum over dim 0
                        if (input->shape().size() == 2 && n->shape().size() == 2 && 
                            input->shape()[0] == 1 && n->shape()[0] > 1) {
                            
                            // 1. Transpose dz: {B, C} -> {C, B}
                            Step t1;
                            t1.op = Op::Transpose;
                            t1.args.push_back(ArgSlot{grad_slot});
                            t1.out_meta = { {n->shape()[1], n->shape()[0]}, n->value.dtype(), n->value.device() };
                            t1.out_slot = plan.num_slots++;
                            plan.steps.push_back(t1);
                            
                            // 2. RowSum: {C, B} -> {C, 1}
                            Step r1;
                            r1.op = Op::RowSum;
                            r1.args.push_back(ArgSlot{t1.out_slot});
                            r1.out_meta = { {n->shape()[1], 1}, n->value.dtype(), n->value.device() };
                            r1.out_slot = plan.num_slots++;
                            plan.steps.push_back(r1);
                            
                            // 3. Transpose back: {C, 1} -> {1, C}
                            Step t2;
                            t2.op = Op::Transpose;
                            t2.args.push_back(ArgSlot{r1.out_slot});
                            t2.out_meta = { {1, n->shape()[1]}, n->value.dtype(), n->value.device() };
                            t2.out_slot = plan.num_slots++;
                            plan.steps.push_back(t2);
                            
                            current_grad_slot = t2.out_slot;
                        }

                        int input_grad_slot;
                        if (grad_slot_of.count(input.get())) {
                            // Accumulate
                            Step st;
                            st.op = Op::Add;
                            st.args.push_back(ArgSlot{grad_slot_of[input.get()]});
                            st.args.push_back(ArgSlot{current_grad_slot});
                            st.out_meta = {input->shape(), input->value.dtype(), input->value.device()};
                            st.out_slot = plan.num_slots++;
                            grad_slot_of[input.get()] = st.out_slot; // Update to new accumulated slot
                            plan.steps.push_back(std::move(st));
                        } else {
                            grad_slot_of[input.get()] = current_grad_slot; 
                        }
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
                    
                    t_op.out_meta = { {y->shape()[1], y->shape()[0]}, y->value.dtype(), y->value.device()};
                    t_op.out_slot = plan.num_slots++;
                    plan.steps.push_back(t_op);
                    
                    st.args.push_back(ArgSlot{t_op.out_slot});
                    st.out_meta = {x->shape(), x->value.dtype(), x->value.device()};
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
                    
                    t_op.out_meta = { {x->shape()[1], x->shape()[0]}, x->value.dtype(), x->value.device()};
                    t_op.out_slot = plan.num_slots++;
                    plan.steps.push_back(t_op);
                    
                    st.args.push_back(ArgSlot{t_op.out_slot});
                    st.args.push_back(ArgSlot{grad_slot});
                    st.out_meta = {y->shape(), y->value.dtype(), y->value.device()};
                    st.out_slot = plan.num_slots++;
                    
                    if (grad_slot_of.count(y)) {
                        Step acc;
                        acc.op = Op::Add;
                        acc.args.push_back(ArgSlot{grad_slot_of[y]});
                        acc.args.push_back(ArgSlot{st.out_slot});
                        acc.out_meta = st.out_meta;
                        acc.out_slot = plan.num_slots++;
                        plan.steps.push_back(std::move(st));
                        plan.steps.push_back(std::move(acc));
                        grad_slot_of[y] = acc.out_slot;
                    } else {
                        plan.steps.push_back(std::move(st));
                        grad_slot_of[y] = st.out_slot;
                    }
                }
            } else if (n->op == Op::GELU) {
                // Identity backward approximation for now
                Node* input = n->inputs[0].get();
                if (input->requires_grad()) {
                     if (grad_slot_of.count(input)) {
                        Step acc;
                        acc.op = Op::Add;
                        acc.args.push_back(ArgSlot{grad_slot_of[input]});
                        acc.args.push_back(ArgSlot{grad_slot});
                        acc.out_meta = {input->shape(), input->value.dtype(), input->value.device()};
                        acc.out_slot = plan.num_slots++;
                        grad_slot_of[input] = acc.out_slot;
                        plan.steps.push_back(std::move(acc));
                    } else {
                        grad_slot_of[input] = grad_slot;
                    }
                }
            } else if (n->op == Op::MSELoss) {
                Node* x = n->inputs[0].get();
                Node* y = n->inputs[1].get();
                int64_t N = x->value.numel();
                
                Step const_step;
                const_step.op = Op::Leaf;
                float scale = 2.0f / N;
                Tensor scale_t = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(x->value.dtype()).with_device(x->value.device()), scale);
                const_step.args.push_back(ArgLit{scale_t});
                const_step.out_meta = { {}, x->value.dtype(), x->value.device() };
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
                
                sub_step.out_meta = {x->shape(), x->value.dtype(), x->value.device()};
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
                    st.out_meta = {x->shape(), x->value.dtype(), x->value.device()};
                    st.out_slot = plan.num_slots++;
                    plan.steps.push_back(std::move(st));
                    grad_slot_of[x] = st.out_slot;
                }
                
                if (y->requires_grad()) {
                    Step neg;
                    neg.op = Op::Leaf; 
                     Tensor neg_one = OwnTensor::Tensor::full(Shape{{1}}, OwnTensor::TensorOptions().with_dtype(y->value.dtype()).with_device(y->value.device()), -1.0f);
                    neg.args.push_back(ArgLit{neg_one});
                    neg.out_meta = { {}, y->value.dtype(), y->value.device() };
                    neg.out_slot = plan.num_slots++;
                    plan.steps.push_back(neg);
                    
                    Step st;
                    st.op = Op::Mul;
                    st.args.push_back(ArgSlot{grad_common});
                    st.args.push_back(ArgSlot{neg.out_slot});
                    st.out_meta = {y->shape(), y->value.dtype(), y->value.device()};
                    st.out_slot = plan.num_slots++;
                    plan.steps.push_back(st);
                    int dy_raw = st.out_slot;

                    Step final_dy;
                    final_dy.op = Op::Mul;
                    final_dy.args.push_back(ArgSlot{dy_raw});
                    final_dy.args.push_back(ArgSlot{grad_slot});
                    final_dy.out_meta = {y->shape(), y->value.dtype(), y->value.device()};
                    final_dy.out_slot = plan.num_slots++;
                    plan.steps.push_back(std::move(final_dy));
                    grad_slot_of[y] = final_dy.out_slot;
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
                st.out_meta = {p->shape(), p->value.dtype(), p->value.device()};
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
    
    if (in_memory_module) {
        try {
            mlir::nova::NovaCompilerAPI compiler;
            mlir::nova::CompilerOptions options;
            options.runFullPipeline = true;
            auto compileResult = compiler.compileString(generated_mlir_opbuilder, "", options);
            if (compileResult.success) {
                generated_mlir_opbuilder = compileResult.output;
                std::cout << "\n=== Optimized MLIR Generated via NovaCompilerAPI ===\n" << generated_mlir_opbuilder <<  std::endl;
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
    
    // Write strides (row-major) - FIX: write strides at correct index positions
    // For an 8x10 tensor: strides[0]=10, strides[1]=1
    int64_t* strides_ptr = reinterpret_cast<int64_t*>(ptr);
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        strides_ptr[i] = stride;  // Write at index i, not at advancing ptr
        stride *= tensor->shape().dims[i];
    }
    
    return descriptor;
}

// Parse a single tensor spec like "8x10xf32" or "f32" (scalar)
// Returns: vector of dimensions (empty for scalar)
std::vector<int64_t> parseTensorSpec(const std::string& tensor_spec) {
    std::vector<int64_t> shape;
    
    // Check if it's a scalar (just "f32" or similar type with no 'x')
    if (tensor_spec.find('x') == std::string::npos) {
        return shape;  // Empty = scalar
    }
    
    // Parse dimensions: "8x10xf32" -> [8, 10]
    size_t pos = 0;
    while (pos < tensor_spec.size()) {
        size_t x_pos = tensor_spec.find('x', pos);
        if (x_pos == std::string::npos) {
            break;  // Last part is the type
        }
        
        std::string dim_str = tensor_spec.substr(pos, x_pos - pos);
        try {
            int64_t dim = std::stoll(dim_str);
            shape.push_back(dim);
        } catch (...) {
            break;
        }
        pos = x_pos + 1;
    }
    return shape;
}

// Parse MLIR function signature to extract ALL result tensor shapes
// Handles: -> (tensor<f32>, tensor<16x10xf32>, tensor<1x10xf32>)
// Returns: vector of shapes (each shape is a vector of dimensions, empty = scalar)
std::vector<std::vector<int64_t>> parseMLIRResultShapes(const std::string& mlir_source) {
    std::vector<std::vector<int64_t>> result_shapes;
    
    // Find the return type section: "-> (" or "-> tensor<"
    size_t arrow_pos = mlir_source.find("-> ");
    if (arrow_pos == std::string::npos) {
        std::cerr << "[ABIAdapter] Warning: Could not find return type in MLIR signature\n";
        return result_shapes;
    }
    
    size_t start = arrow_pos + 3;  // After "-> "
    
    // Check if it's a tuple of results: "-> (tensor<...>, tensor<...>)"
    if (mlir_source[start] == '(') {
        // Multiple results - find matching closing paren
        size_t open_paren = start;
        size_t close_paren = mlir_source.find(')', open_paren);
        if (close_paren == std::string::npos) {
            std::cerr << "[ABIAdapter] Warning: Malformed tuple return type\n";
            return result_shapes;
        }
        
        std::string tuple_content = mlir_source.substr(open_paren + 1, close_paren - open_paren - 1);
        
        // Parse each "tensor<...>" in the tuple
        size_t pos = 0;
        while (pos < tuple_content.size()) {
            // Skip whitespace
            while (pos < tuple_content.size() && (tuple_content[pos] == ' ' || tuple_content[pos] == ',')) {
                pos++;
            }
            if (pos >= tuple_content.size()) break;
            
            // Find "tensor<"
            size_t tensor_start = tuple_content.find("tensor<", pos);
            if (tensor_start == std::string::npos) break;
            
            size_t spec_start = tensor_start + 7;  // After "tensor<"
            size_t spec_end = tuple_content.find('>', spec_start);
            if (spec_end == std::string::npos) break;
            
            std::string spec = tuple_content.substr(spec_start, spec_end - spec_start);
            result_shapes.push_back(parseTensorSpec(spec));
            
            pos = spec_end + 1;
        }
    } else if (mlir_source.substr(start, 7) == "tensor<") {
        // Single result: "-> tensor<...>"
        size_t spec_start = start + 7;
        size_t spec_end = mlir_source.find('>', spec_start);
        if (spec_end != std::string::npos) {
            std::string spec = mlir_source.substr(spec_start, spec_end - spec_start);
            result_shapes.push_back(parseTensorSpec(spec));
        }
    }
    
    std::cout << "[ABIAdapter] Parsed " << result_shapes.size() << " result shapes\n";
    return result_shapes;
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
    
    // Build memref descriptors for all input arguments
    std::vector<std::vector<char>> input_descriptors;
    input_descriptors.reserve(total_args);
    
    // Build descriptors for inputs
    for (size_t i = 0; i < num_inputs; ++i) {
        auto* tensor = static_cast<Tensor*>(args[i]);
        input_descriptors.push_back(buildMemRefDescriptor(tensor, plan.sig.in_meta[i]));
        
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
        input_descriptors.push_back(buildMemRefDescriptor(tensor, plan.sig.param_meta[i]));
        
        std::cout << "[ABIAdapter]   Param[" << i << "]: shape=(";
        for (size_t j = 0; j < plan.sig.param_meta[i].shape.size(); ++j) {
            std::cout << plan.sig.param_meta[i].shape[j];
            if (j < plan.sig.param_meta[i].shape.size() - 1) std::cout << "x";
        }
        std::cout << ")\n";
    }
    
    // Parse MLIR signature to get ALL result shapes
    std::vector<std::vector<int64_t>> result_shapes = parseMLIRResultShapes(g_aot_context.mlir_source);
    size_t num_results = result_shapes.size();
    
    std::cout << "[ABIAdapter] Number of results: " << num_results << "\n";
    
    // Calculate the size of each result's memref descriptor and the total packed struct size
    // MLIR memref descriptor layout:
    // - Scalar (rank 0): { ptr, ptr, i64 } = 2 pointers + 1 i64
    // - Rank N: { ptr, ptr, i64, array<N x i64>, array<N x i64> } = 2 pointers + 1 + 2*N i64s
    std::vector<size_t> result_desc_sizes;
    std::vector<size_t> result_desc_offsets;
    size_t packed_struct_size = 0;
    
    for (size_t r = 0; r < num_results; ++r) {
        size_t result_rank = result_shapes[r].size();
        size_t desc_size;
        
        if (result_rank == 0) {
            // Scalar: 2 pointers + 1 offset
            desc_size = sizeof(void*) * 2 + sizeof(int64_t);
        } else {
            // Non-scalar: 2 pointers + offset + N sizes + N strides
            desc_size = sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * result_rank);
        }
        
        result_desc_sizes.push_back(desc_size);
        result_desc_offsets.push_back(packed_struct_size);
        packed_struct_size += desc_size;
        
        std::cout << "[ABIAdapter]   Result[" << r << "]: rank=" << result_rank 
                  << ", desc_size=" << desc_size << ", offset=" << result_desc_offsets[r] << "\n";
    }
    
    std::cout << "[ABIAdapter] Total packed output struct size: " << packed_struct_size << " bytes\n";
    
    // Allocate the packed output struct and result data buffers
    std::vector<char> packed_output(packed_struct_size, 0);
    std::vector<void*> result_buffers(num_results);
    
    // Initialize each result memref descriptor within the packed struct
    for (size_t r = 0; r < num_results; ++r) {
        size_t result_rank = result_shapes[r].size();
        char* desc_ptr = packed_output.data() + result_desc_offsets[r];
        
        // Allocate buffer for result data
        size_t buffer_size = sizeof(float);
        for (auto dim : result_shapes[r]) {
            buffer_size *= dim;
        }
        if (buffer_size == 0) buffer_size = sizeof(float);  // At least 1 element for scalar
        
        result_buffers[r] = std::malloc(buffer_size);
        std::memset(result_buffers[r], 0, buffer_size);
        
        // Set allocated and aligned pointers
        *reinterpret_cast<void**>(desc_ptr) = result_buffers[r];
        *reinterpret_cast<void**>(desc_ptr + sizeof(void*)) = result_buffers[r];
        *reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2) = 0;  // offset
        
        if (result_rank > 0) {
            // Fill in sizes and strides for non-scalar results
            int64_t* sizes_ptr = reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2 + sizeof(int64_t));
            int64_t* strides_ptr = sizes_ptr + result_rank;
            
            for (size_t i = 0; i < result_rank; ++i) {
                sizes_ptr[i] = result_shapes[r][i];
            }
            
            int64_t stride = 1;
            for (int i = result_rank - 1; i >= 0; --i) {
                strides_ptr[i] = stride;
                stride *= result_shapes[r][i];
            }
        }
    }
    
    // Build the argument array for the C interface call
    // Order: 1 packed output struct pointer, then N input memref pointers
    std::vector<void*> call_args;
    call_args.push_back(packed_output.data());  // Packed output struct
    for (auto& desc : input_descriptors) {
        call_args.push_back(desc.data());
    }
    
    std::cout << "[ABIAdapter] Calling compiled function with " << call_args.size() << " args (1 output + " 
              << total_args << " inputs)...\n";
    
    // Call the compiled function using libffi for dynamic argument handling
    // Since we don't have libffi set up for this, use a switch-case approach
    // The function signature is: void func(void* output, void* in1, void* in2, ...)
    
    // Pad to 10 elements for safety
    while (call_args.size() < 10) {
        call_args.push_back(nullptr);
    }
    
    using CIfaceFunc = void (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
    
    reinterpret_cast<CIfaceFunc>(g_aot_context.ciface_func)(
        call_args[0], call_args[1], call_args[2], call_args[3], call_args[4],
        call_args[5], call_args[6], call_args[7], call_args[8], call_args[9]
    );
    
    std::cout << "[ABIAdapter] Function returned\n";
    
    // Extract ALL results from the packed output struct and create output tensors
    auto* output_tensors = new std::vector<Tensor>();
    output_tensors->reserve(num_results);
    
    for (size_t r = 0; r < num_results; ++r) {
        size_t result_rank = result_shapes[r].size();
        char* desc_ptr = packed_output.data() + result_desc_offsets[r];
        
        // Read the memref descriptor fields
        void* aligned = *reinterpret_cast<void**>(desc_ptr + sizeof(void*));
        int64_t offset = *reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2);
        
        std::cout << "[ABIAdapter] Extracting result " << r << " aligned=" << aligned << " offset=" << offset << "\n";
        
        if (!aligned) {
            std::cerr << "[ABIAdapter] Error: Invalid result " << r << " from compiled function\n";
            continue;
        }
        
        if (result_rank == 0) {
            // Scalar result
            Tensor out(OwnTensor::Shape{{1}}, OwnTensor::Dtype::Float32,
                       OwnTensor::DeviceIndex(OwnTensor::Device::CPU), false);
            float* data_ptr = reinterpret_cast<float*>(aligned) + offset;
            float scalar_val = *data_ptr;
            out.data<float>()[0] = scalar_val;
            
            std::cout << "[ABIAdapter]   Result[" << r << "]: scalar = " << scalar_val << "\n";
            output_tensors->push_back(std::move(out));
        } else {
            // Non-scalar result - read sizes from descriptor
            int64_t* sizes_ptr = reinterpret_cast<int64_t*>(desc_ptr + sizeof(void*) * 2 + sizeof(int64_t));
            std::vector<int64_t> actual_shape(sizes_ptr, sizes_ptr + result_rank);
            
            std::cout << "[ABIAdapter]   Result[" << r << "]: shape=(";
            for (size_t d = 0; d < result_rank; ++d) {
                std::cout << actual_shape[d];
                if (d < result_rank - 1) std::cout << "x";
            }
            std::cout << ")\n";
            
            Tensor out(OwnTensor::Shape{actual_shape}, OwnTensor::Dtype::Float32,
                       OwnTensor::DeviceIndex(OwnTensor::Device::CPU), false);
            
            size_t total_elements = 1;
            for (auto dim : actual_shape) {
                total_elements *= dim;
            }
            
            float* data_ptr = reinterpret_cast<float*>(aligned) + offset;
            std::memcpy(out.data<float>(), data_ptr, total_elements * sizeof(float));
            output_tensors->push_back(std::move(out));
        }
        
        // Free the result buffer
        std::free(result_buffers[r]);
    }
    
    std::cout << "[ABIAdapter] Successfully executed, returning " << output_tensors->size() << " tensors\n";
    
    return output_tensors;
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
    return p->run(inputs, params, out);
}

bool Compiled::run(const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& params,
                   std::vector<Tensor>& outs) const {
    // Check if we have a pre-compiled function from compile()
    if (!compiled_func) {
        // Fallback to interpreter if compilation failed or wasn't attempted
        std::cout << "[Run] No compiled function available, using interpreter...\n";
        return p->run(inputs, params, outs);
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
    
    // Call the ABI adapter directly - now returns vector<Tensor>*
    auto* result_tensors = static_cast<std::vector<Tensor>*>(ABIAdapter(exec_inputs.data()));
    
    if (!result_tensors || result_tensors->empty()) {
        std::cerr << "[Run] Error: ABI adapter returned null or empty\n";
        delete result_tensors;
        return false;
    }

    // Move all results to output vector
    std::cout << "[Run] ABI adapter returned " << result_tensors->size() << " tensors\n";
    for (auto& tensor : *result_tensors) {
        outs.push_back(std::move(tensor));
    }
    delete result_tensors;
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
