//===- ModArithBase.td - Base definitions for mod_arith -----*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MODARITH_IR_MODARITH_BASE_TD
#define MLIR_DIALECT_MODARITH_IR_MODARITH_BASE_TD

include "mlir/IR/OpBase.td"

def ModArith_Dialect : Dialect {
  let name = "mod_arith";
  let cppNamespace = "::mlir::mod_arith";
  let description = [{
    The `mod_arith` dialect provides types and operations for modular arithmetic.

    // FIXME: expand with more details and examples from the RFC
  }];
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

#endif // MLIR_DIALECT_MODARITH_IR_MODARITH_BASE_TD
