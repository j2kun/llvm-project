//===- InitTensorToAllocTensor.cpp - Lower tensor.empty to alloc_tensor ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_EMPTYTENSORTOALLOCTENSORPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace {
struct EmptyTensorLoweringPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes());
    return success();
  }
};

struct EmptyTensorToAllocTensor
    : public bufferization::impl::EmptyTensorToAllocTensorPassBase<
          EmptyTensorToAllocTensor> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tensor::TensorDialect, bufferization::BufferizationDialect>();
  }
};
} // namespace

void bufferization::populateEmptyTensorToAllocTensorPattern(
    RewritePatternSet &patterns) {
  patterns.insert<EmptyTensorLoweringPattern>(patterns.getContext());
}

void EmptyTensorToAllocTensor::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  populateEmptyTensorToAllocTensorPattern(patterns);
  if (failed(applyPatternsGreedily(op, std::move(patterns))))
    signalPassFailure();
}
