// RUN: mlir-tblgen -gen-pipeline-decls -I %S/../../include -name Test %s | FileCheck %s

include "mlir/Pass/PipelineBase.td"

// CHECK: namespace mlir {

def SimpleOptimize : Pipeline<"simple-opt", "func.func"> {
  let summary = "Simple optimization pipeline";
  let description = "Basic canonicalization and CSE";
  
  let options = [
    Option<"level", "level", "int", "1", "Optimization level">
  ];
  
  let elements = [
    PassElement<"canonicalize">,
    PassElement<"cse">
  ];
}

// CHECK: struct SimpleOptimizeOptions {
// CHECK:   int level = 1;
// CHECK: };

// CHECK: void buildSimpleOptimizePipeline(::mlir::OpPassManager &pm, const SimpleOptimizeOptions &options);

// CHECK: void registerTestPipelines();

// CHECK: } // namespace mlir