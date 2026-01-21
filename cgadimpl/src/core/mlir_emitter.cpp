#include "ad/core/mlir_emitter.hpp"
#include "ad/core/graph.hpp"
#include "ad/core/schema.hpp"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"

namespace ag::jit {

MLIREmitter::MLIREmitter() {
    context_ = std::make_shared<mlir::MLIRContext>();
    registerDialects();
}

MLIREmitter::~MLIREmitter() = default;

void MLIREmitter::registerDialects() {
    // The emitter only needs Nova and Func dialects to build the initial IR.
    // Optimization and bufferization are now handled within NovaCompilerAPI.
    context_->getOrLoadDialect<mlir::nova::NovaDialect>();
    context_->getOrLoadDialect<mlir::func::FuncDialect>();
}

mlir::Type MLIREmitter::dtypeToMLIRType(mlir::OpBuilder& builder, OwnTensor::Dtype dtype) {
    switch (dtype) {
        case OwnTensor::Dtype::Float32:  return builder.getF32Type();
        case OwnTensor::Dtype::Float16:  return builder.getF16Type();
        case OwnTensor::Dtype::Bfloat16: return builder.getBF16Type();
        case OwnTensor::Dtype::Int32:    return builder.getI32Type();
        case OwnTensor::Dtype::Int64:    return builder.getI64Type();
        default:
            llvm::errs() << "Unsupported dtype in MLIR emission\n";
            return builder.getF32Type(); // fallback
    }
}

mlir::RankedTensorType MLIREmitter::createTensorType(
    mlir::OpBuilder& builder,
    const std::vector<int64_t>& shape,
    OwnTensor::Dtype dtype,
    OwnTensor::DeviceIndex device
) {
    auto elemType = dtypeToMLIRType(builder, dtype);
    if (device.is_cuda()) {
        std::string deviceStr = std::to_string(device.index);
        auto deviceAttr = ::mlir::nova::NovaDeviceAttr::get(
            builder.getContext(), 
            builder.getStringAttr(deviceStr)
        );
        return mlir::RankedTensorType::get(shape, elemType, deviceAttr);
    }
return mlir::RankedTensorType::get(shape, elemType);

}
// mlir::RankedTensorType MLIREmitter::createTensorType1(
//     mlir::OpBuilder& builder,
//     const std::vector<int64_t>& shape,
//     OwnTensor::Dtype dtype, 
//     mlir::nova::NovaDeviceAttr device
// ) {
//     auto elemType = dtypeToMLIRType(builder, dtype);
//     return  mlir::RankedTensorType::get(
//       shape, elemType,
//       NovaDeviceAttr::get(builder.getContext(), builder.getStringAttr(device)));
// }

std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::string> 
MLIREmitter::emitModule(const Plan& plan) {
    mlir::OpBuilder builder(context_.get());
    auto loc = builder.getUnknownLoc();

    // Create module
    auto module = mlir::ModuleOp::create(loc);
    module->setAttr("nova.num_inputs", builder.getI64IntegerAttr(plan.sig.in_meta.size()));
    builder.setInsertionPointToEnd(module.getBody());

    // Build function signature
    llvm::SmallVector<mlir::Type, 8> inputTypes;
    
    // Add input tensors to function signature
    for (const auto& meta : plan.sig.in_meta) {
        inputTypes.push_back(createTensorType(builder, meta.shape, meta.dtype, meta.device));
    }
    
    // Add parameter tensors to function signature
    for (const auto& meta : plan.sig.param_meta) {
        inputTypes.push_back(createTensorType(builder, meta.shape, meta.dtype, meta.device));
    }

    // Output types
    llvm::SmallVector<mlir::Type, 4> outputTypes;
    
    // Build a map from slot to meta for quick lookup
    llvm::DenseMap<int, TensorMetadata> slotMetaMap;
    for (const auto& step : plan.steps) {
        slotMetaMap[step.out_slot] = step.out_meta;
    }

    for (int slot : plan.out_slots) {
        if (slotMetaMap.find(slot) == slotMetaMap.end()) {
            llvm::errs() << "Error: Output slot " << slot << " metadata not found\n";
            return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
        }
        const auto& meta = slotMetaMap[slot];
        auto shape = meta.shape;
        
        outputTypes.push_back(createTensorType(builder, meta.shape, meta.dtype, meta.device));
    }

    // Create function type
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    // Create function
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    
    // Add llvm.emit_c_interface attribute to generate C-compatible wrapper
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Map slots to MLIR values
    llvm::DenseMap<int, mlir::Value> slotMap;

    // Map function arguments to their sources
    size_t argIdx = 0;
    
    // First, map all inputs
    for (size_t i = 0; i < plan.sig.in_meta.size(); ++i) {
        // Inputs will be referenced via ArgInput in steps
        // We'll handle them during step processing
        argIdx++;
    }
    
    // Then params
    for (size_t i = 0; i < plan.sig.param_meta.size(); ++i) {
        argIdx++;
    }

    // Helper to get MLIR value from Arg
    auto getValueForArg = [&](const Arg& arg) -> mlir::Value {
        return std::visit([&](auto&& a) -> mlir::Value {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, ArgInput>) {
                return entryBlock.getArgument(a.idx);
            } else if constexpr (std::is_same_v<T, ArgParam>) {
                return entryBlock.getArgument(plan.sig.in_meta.size() + a.idx);
            } else if constexpr (std::is_same_v<T, ArgSlot>) {
                return slotMap[a.slot];
            } else if constexpr (std::is_same_v<T, ArgLit>) {
                // For literals, we create a nova.constant op
                const auto& t = a.t;
                auto shape = t.shape().dims;
                auto type = createTensorType(builder, shape, t.dtype(), t.device());
                
                // For now, support float32 literals
                if (t.dtype() == OwnTensor::Dtype::Float32) {
                    auto data = t.to_cpu();
                    auto attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(static_cast<const float*>(data.data()), t.numel()));
                    return builder.create<mlir::nova::ConstantOp>(loc, type, attr).getResult();
                } else {
                    llvm::errs() << "Error: Unsupported literal dtype in MLIR emission\n";
                    return mlir::Value();
                }
            }
        }, arg);
    };

    // Process each step
    for (const auto& step : plan.steps) {
        llvm::SmallVector<mlir::Value, 4> operands;
        
        // Gather operands
        for (const auto& arg : step.args) {
            auto val = getValueForArg(arg);
            if (!val) {
                llvm::errs() << "Error: Failed to get value for argument\n";
                return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
            }
            operands.push_back(val);
        }

        // Create the appropriate Nova operation
        mlir::Value result;
        auto resultType = createTensorType(builder, step.out_meta.shape, step.out_meta.dtype, step.out_meta.device);

        switch (step.op) {
            case Op::Add:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::AddOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::Sub:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::SubOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::Mul:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::MulOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::MatMul:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::MatmulOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::Exp:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::ExpOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Log:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::LogOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Tanh:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::TanhOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Sign:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::SignOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Transpose:
                if (operands.size() == 1) {
                    // Uses default axes (-1, -2)
                    result = builder.create<mlir::nova::TransposeOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Relu:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::ReluOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::GELU:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::GeluOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Leaf:
                if (operands.size() == 1) {
                    // Leaf just passes through the literal value (which was created as a constant in getValueForArg)
                    result = operands[0];
                }
                break;



            case Op::Sum: {
                if (operands.size() == 1) {
                    // Sum all dimensions - Nova expects rank-0 for total reduction if keepdims=false
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::SUM,
                        operands[0],
                        resultType,
                        /*keepdims=*/false,
                        /*dimension=*/llvm::ArrayRef<int64_t>{},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;
            }

            case Op::RowSum: {
                if (operands.size() == 1) {
                    auto inputType = cast<mlir::RankedTensorType>(operands[0].getType());
                    auto rowSumType = mlir::RankedTensorType::get(resultType.getShape(), resultType.getElementType(), inputType.getEncoding());
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::SUM,
                        operands[0],
                        rowSumType,
                        /*keepdims=*/true,
                        llvm::ArrayRef<int64_t>{1},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;
            }

            case Op::RowMax: {
                if (operands.size() == 1) {
                    auto inputType = cast<mlir::RankedTensorType>(operands[0].getType());
                    auto rowMaxType = mlir::RankedTensorType::get(resultType.getShape(), resultType.getElementType(), inputType.getEncoding());
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::MAX,
                        operands[0],
                        rowMaxType,
                        /*keepdims=*/true,
                        llvm::ArrayRef<int64_t>{1},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;
            }

            case Op::MeanAll: {
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::MEAN,
                        operands[0],
                        resultType,
                        /*keepdims=*/false,
                        llvm::ArrayRef<int64_t>{},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;
            }

            case Op::MSELoss: {
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::MseOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;
            }

            case Op::MAELoss: {
                if (operands.size() == 2) {
                    auto inputType = cast<mlir::RankedTensorType>(operands[0].getType());
                    result = builder.create<mlir::nova::MaeOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;
            }

            case Op::SoftmaxRow:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::SoftmaxOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::BinaryCrossEntropy: {
                if (operands.size() == 2) {
                    auto inputType = cast<mlir::RankedTensorType>(operands[0].getType());
                    result = builder.create<mlir::nova::BceOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;
            }
            case Op::CategoricalCrossEntropy: {
                if (operands.size() == 2) {
                    auto inputType = cast<mlir::RankedTensorType>(operands[0].getType());
                    result = builder.create<mlir::nova::CceOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;
            }

            case Op::CeWithLogits:
                llvm::errs() << "Warning: CeWithLogits not yet supported in MLIR emitter\n";
                return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
                break;

            case Op::KLDivergence:
                llvm::errs() << "Warning: KLDivergence not yet supported in MLIR emitter\n";
                return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
                break;

            case Op::SparseCeWithLogits: {
                if (operands.size() == 2) {
                    auto inputType = cast<mlir::RankedTensorType>(operands[0].getType());
                    result = builder.create<mlir::nova::SceOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;
            }

            default:
                llvm::errs() << "Warning: Unsupported op " << op_name(step.op) 
                           << " in MLIR emission, skipping\n";
                // For unsupported ops, we can't continue
                return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
        }

        if (!result) {
            llvm::errs() << "Error: Failed to create MLIR op for " << op_name(step.op) << "\n";
            return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
        }

        // Store result in slot map
        slotMap[step.out_slot] = result;
    }

    // Create return statement
    // Create return statement
    llvm::SmallVector<mlir::Value, 4> returnValues;
    for (int slot : plan.out_slots) {
        auto val = slotMap[slot];
        if (!val) {
            llvm::errs() << "Error: Output slot " << slot << " not found in MLIR emission\n";
            return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
        }
        returnValues.push_back(val);
    }

    builder.create<mlir::func::ReturnOp>(loc, returnValues);

    // Verify the module
    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "Error: MLIR module verification failed\n";
        module.dump();
        return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
    }

    // Serialize to string
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.print(os);
    os.flush();

    return {mlir::OwningOpRef<mlir::ModuleOp>(module), mlirStr};
}

} // namespace ag::jit