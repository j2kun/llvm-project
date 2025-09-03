// RUN: mlir-tblgen -gen-pipeline-doc -I %S/../../include %s | FileCheck %s

include "mlir/Pass/PipelineBase.td"

def OptimizePipeline : Pipeline<"optimize", "builtin.module"> {
  let summary = "Full optimization pipeline";
  let description = [{
    Comprehensive optimization including canonicalization, inlining, and cleanup.
  }];
  
  let options = [
    Option<"enableInlining", "enable-inlining", "bool", "true", "Enable inlining">
  ];
  
  let elements = [
    PassElement<"canonicalize">,
    PassElement<"inline">,
    PassElement<"symbol-dce">
  ];
}

// CHECK: # Generated Pipeline Documentation

// CHECK: ## optimize

// CHECK: Full optimization pipeline

// CHECK: Comprehensive optimization including canonicalization, inlining, and cleanup.

// CHECK: ### Options

// CHECK: - `--enable-inlining`: Enable inlining