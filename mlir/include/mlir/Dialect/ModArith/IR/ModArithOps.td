//===- ModArithOps.td - ModArith op definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MODARITH_IR_MODARITH_OPS_TD
#define MLIR_DIALECT_MODARITH_IR_MODARITH_OPS_TD

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/ModArith/IR/ModArithBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for mod_arith dialect ops. Ops in this dialect have no side
// effects and can be applied element-wise to vectors and tensors.
class ModArith_Op<string mnemonic, list<Trait> traits = []> :
    Op<
      ModArith_Dialect,
      mnemonic,
      traits # [Pure] # ElementwiseMappable.traits
    >;

// Base class for binary mod_arith operations.
class ModArith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    ModArith_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$lhs, SignlessIntegerOrIndexLike:$rhs);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}


#endif // MLIR_DIALECT_MODARITH_IR_MODARITH_OPS_TD
