//===- pipelines-test.td - Test pipeline definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.  
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test pipeline definitions for the pipeline generation
// system.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PipelineBase.td"

// Simple function optimization pipeline
def FuncOptimize : Pipeline<"func-opt", "func.func"> {
  let summary = "Standard function-level optimizations";
  let description = [{
    Applies canonicalization, CSE, and other function-level optimizations.
  }];
  
  let options = [
    Option<"maxIterations", "max-iter", "int", "3", 
           "Maximum number of canonicalization iterations">
  ];
  
  let elements = [
    PassElement<"canonicalize"> {
      let options = (ins "max-iterations", "$maxIterations");
    },
    PassElement<"cse">,
    PassElement<"sccp">
  ];
}

// Main optimization pipeline
def OptimizationPipeline : Pipeline<"optimize", "builtin.module"> {
  let summary = "Full program optimization pipeline";
  let description = [{
    Comprehensive optimization pipeline including function-level optimizations,
    inlining, and symbol cleanup.
  }];
  
  let options = [
    Option<"enableInlining", "enable-inlining", "bool", "true",
           "Enable aggressive inlining">,
    Option<"funcOptLevel", "func-opt-level", "int", "2",
           "Function optimization level">
  ];
  
  let elements = [
    // Apply function optimizations
    NestedElement<"func.func"> {
      let elements = [
        PipelineRef<"func-opt"> {
          let options = (ins "max-iter", "$funcOptLevel");
        }
      ];
    },
    
    // Module-level passes
    PassElement<"symbol-dce">,
    
    // Conditional inlining
    PassElement<"inline"> {
      let options = (ins "default-pipeline", "");
    },
    
    // Re-run function opts after inlining
    NestedElement<"func.func"> {
      let elements = [
        PipelineRef<"func-opt"> {
          let options = (ins "max-iter", "1");
        }
      ];
    }
  ];
}

// GPU-specific pipeline
def GPUPipeline : Pipeline<"gpu-opt", "builtin.module"> {
  let summary = "GPU-specific optimization pipeline";
  
  let options = [
    Option<"targetGPU", "target", "string", "\"generic\"",
           "Target GPU architecture">
  ];
  
  let elements = [
    NestedElement<"gpu.module"> {
      let elements = [
        PassElement<"gpu-kernel-outlining">,
        NestedElement<"func.func"> {
          let elements = [
            PassElement<"canonicalize">,
            PassElement<"gpu-launch-sink-index-computations">
          ];
        }
      ];
    },
    PassElement<"gpu-async-region">,
    PassElement<"convert-gpu-to-nvvm"> {
      let options = (ins "index-bitwidth", "64");
    }
  ];
}