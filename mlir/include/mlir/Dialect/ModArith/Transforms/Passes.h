//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MODARITH_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_MODARITH_TRANSFORMS_PASSES_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mod_arith {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/ModArith/Transforms/Passes.h.inc"
} // namespace mod_arith
} // namespace mlir

#endif // MLIR_DIALECT_MODARITH_TRANSFORMS_PASSES_H_
