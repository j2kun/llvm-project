//===-- PipelineBase.td - Base pipeline definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for defining pipeline registration and other
// mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PIPELINEBASE
#define MLIR_PASS_PIPELINEBASE

include "mlir/Pass/PassBase.td"
include "mlir/IR/Utils.td"

//===----------------------------------------------------------------------===//
// Pipeline Elements
//===----------------------------------------------------------------------===//

// Base class for all pipeline elements
class PipelineElement;

// Reference to a pass with options
class PassElement<string passName> : PipelineElement {
  // The pass to instantiate (must match registered pass argument)
  string pass = passName;
  
  // Options to pass to this pass instance as a list of strings
  // Format: ["option_name", "option_value", "option_name", "option_value", ...]
  list<string> options = [];
}

// Nested pipeline for a different operation type
class NestedElement<string targetOp> : PipelineElement {
  // The operation type this nested pipeline targets
  string operation = targetOp;
  
  // Sequence of elements in this nested pipeline
  list<PipelineElement> elements = [];
}

// Reference to a pipeline with options
class PipelineRef<string pipelineName> : PipelineElement {
  // The pipeline to instantiate
  string pipeline = pipelineName;
  
  // Options to pass to the pipeline as a list of strings
  list<string> options = [];
}

//===----------------------------------------------------------------------===//
// Pipeline Definitions
//===----------------------------------------------------------------------===//

// Main pipeline definition class
class Pipeline<string pipelineArg, string targetOp = ""> {
  // Command line argument for the pipeline
  string argument = pipelineArg;
  
  // Target operation type (empty means any/module-level)
  string operation = targetOp;
  
  // Short summary of the pipeline
  string summary = "";
  
  // Detailed description
  string description = "";
  
  // Pipeline-level options
  list<Option> options = [];
  
  // Sequence of pipeline elements
  list<PipelineElement> elements = [];
}

#endif // MLIR_PASS_PIPELINEBASE