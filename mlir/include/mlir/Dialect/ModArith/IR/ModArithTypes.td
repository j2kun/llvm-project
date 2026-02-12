//===- ModArithTypes.td - ModArith types -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MODARITH_IR_MODARITH_TYPES_TD
#define MLIR_DIALECT_MODARITH_IR_MODARITH_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/ModArith/IR/ModArithBase.td"

//===----------------------------------------------------------------------===//
// ModArith Types
//===----------------------------------------------------------------------===//

class ModArith_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ModArith_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


#endif // MLIR_DIALECT_MODARITH_IR_MODARITH_TYPES_TD
