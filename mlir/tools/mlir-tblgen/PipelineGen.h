//===- PipelineGen.h - MLIR pipeline registration generator ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PipelineGen uses the description of pipelines to generate pipeline
// registration and builder functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIR_TBLGEN_PIPELINEGEN_H_
#define MLIR_TOOLS_MLIR_TBLGEN_PIPELINEGEN_H_

#include "llvm/Support/raw_ostream.h"

namespace llvm {
class RecordKeeper;
}

namespace mlir {

/// Emit pipeline declarations.
void emitPipelineDecls(const llvm::RecordKeeper &recordKeeper,
                       llvm::raw_ostream &os, llvm::StringRef groupName);

/// Emit pipeline implementations.  
void emitPipelineImpls(const llvm::RecordKeeper &recordKeeper,
                       llvm::raw_ostream &os, llvm::StringRef groupName);

/// Emit pipeline documentation.
void emitPipelineDoc(const llvm::RecordKeeper &recordKeeper,
                     llvm::raw_ostream &os);

} // namespace mlir

#endif // MLIR_TOOLS_MLIR_TBLGEN_PIPELINEGEN_H_