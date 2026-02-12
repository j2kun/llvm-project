//===- ModArithTypes.td - ModArith types -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MODARITH_IR_MODARITH_TYPES_TD
#define MLIR_DIALECT_MODARITH_IR_MODARITH_TYPES_TD

include "mlir/Dialect/ModArith/IR/ModArithBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// ModArith Types
//===----------------------------------------------------------------------===//

class ModArith_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ModArith_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ModArith_ModArithType : ModArith_Type<"ModArith", "int", [MemRefElementTypeInterface, OpAsmTypeInterface]> {
  let summary = "Integer type with modular arithmetic";
  let description = [{
    `mod_arith.int<n>` represents an element of the ring of integers modulo `n`.
    The `modulus` attribute is the ring modulus, and it is specified with an
    integer type, for example, `mod_arith.int<65537 : i32>`. This corresponds
    to the storage type for the modulus, and is `i64` by default.

    It is required that the underlying integer type should be larger than
    twice the modulus (have one extra bit of storage space) to avoid signedness
    issues. For example, when `modulus == 2 ** 32 - 1`, the underlying type
    for the modulus should be at least `i33`, though `i64` is a natural choice.

    Passes may allow intermediate values that do not always produce a
    canonical representative in `[0, modulus)`. For example, if the machine storage
    type is `i64`, but the `modulus` fits within an `i32`, a lowering could
    allow intermediate arithmetic values to grow to as large as an `i64` before
    reducing them. However, all passes must ensure that values used outside
    the local scope (e.g., function return values or arguments to calls to linked
    functions) are appropriately reduced to the canonical representative.
    Given these guarantees, static analysis can be used to determine the
    possible range in which a canonical representative could be.

    Examples:

    ```
    !Zp1 = !mod_arith.int<7> // with an implicit storage type of i64
    !Zp2 = !mod_arith.int<65537 : i32>
    !Zp3 = !mod_arith.int<536903681 : i64>
    ```
  }];

  let parameters = (ins "::mlir::IntegerAttr":$modulus);
  let assemblyFormat = "`<` $modulus `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // OpAsmTypeInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "Z";
      os << getModulus().getValue();
      os << "_";
      os << getModulus().getType();
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def ModArithLike: TypeOrValueSemanticsContainer<ModArith_ModArithType, "mod_arith-like">;

#endif // MLIR_DIALECT_MODARITH_IR_MODARITH_TYPES_TD
