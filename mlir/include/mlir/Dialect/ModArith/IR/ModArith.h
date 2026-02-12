//===- ModArith.h - ModArith dialect ------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MODARITH_IR_MODARITH_H_
#define MLIR_DIALECT_MODARITH_IR_MODARITH_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// ModArith Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ModArith/IR/ModArithOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// ModArith Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/ModArith/IR/ModArithOps.h.inc"

#endif // MLIR_DIALECT_MODARITH_IR_MODARITH_H_
