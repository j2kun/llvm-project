//===- PolynomialMystery.cpp - Canonicalize MLIR operations
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"
#include <iostream>

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialOps.h"

namespace mlir {
#define GEN_PASS_DEF_POLYNOMIALMYSTERY
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::polynomial;

/// Canonicalize operations in nested regions.
struct PolynomialMystery
    : public impl::PolynomialMysteryBase<PolynomialMystery> {
  PolynomialMystery() = default;

  void runOnOperation() override {
    // pass
    std::cerr << "Before conversion\n";
    getOperation()->walk([&](ConstantOp op) {
      if (auto attr = dyn_cast<TypedIntPolynomialAttr>(op.getValue())) {
        for (const IntMonomial& term : attr.getValue().getPolynomial().getTerms()) {
          term.getCoefficient().dump();
          term.getExponent().dump();
        }
      }
    });
  }
};
