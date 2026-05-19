//===- Pipeline.h - TableGen Pipeline Record Wrapper ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Pipeline class, which is a wrapper around a TableGen
// Record representing a Pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PIPELINE_H_
#define MLIR_TABLEGEN_PIPELINE_H_

#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class DagInit;
class Init;
class ListInit;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Wrapper class around a TableGen Pipeline record.
class Pipeline {
public:
  explicit Pipeline(const llvm::Record *def);

  /// Get the command line argument for this pipeline.
  llvm::StringRef getArgument() const;

  /// Get the target operation for this pipeline.
  llvm::StringRef getOperation() const;

  /// Get the summary for this pipeline.
  llvm::StringRef getSummary() const;

  /// Get the description for this pipeline.
  llvm::StringRef getDescription() const;

  /// Get the pipeline options.
  std::vector<PassOption> getOptions() const;

  /// Get the pipeline elements as a list.
  const llvm::ListInit *getElements() const;

  /// Get the underlying TableGen record.
  const llvm::Record *getDef() const { return def; }

private:
  /// The TableGen record this pipeline is constructed from.
  const llvm::Record *def;
};

/// Wrapper class around a TableGen PipelineElement record.
class PipelineElement {
public:
  explicit PipelineElement(const llvm::Init *init);

  /// Check if this is a pass element.
  bool isPassElement() const;

  /// Check if this is a nested element.
  bool isNestedElement() const;

  /// Check if this is a pipeline reference element.
  bool isPipelineElement() const;

  /// Get the pass name (for PassElement).
  llvm::StringRef getPassName() const;

  /// Get the target operation (for NestedElement).
  llvm::StringRef getOperation() const;

  /// Get the pipeline name (for PipelineElement).
  llvm::StringRef getPipelineName() const;

  /// Get the nested elements (for NestedElement).
  const llvm::ListInit *getElements() const;

  /// Get the options list.
  const llvm::ListInit *getOptions() const;

  /// Get the underlying TableGen record.
  const llvm::Record *getDef() const { return def; }

private:
  /// The TableGen record this element is constructed from.
  const llvm::Record *def;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PIPELINE_H_