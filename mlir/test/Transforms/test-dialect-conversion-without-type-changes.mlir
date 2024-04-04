// RUN: mlir-opt -allow-unregistered-dialect -test-dialect-conversion-without-type-changes -verify-diagnostics %s | FileCheck %s

// Test that SSA values are properly replaced in dialect conversion even when
// types are not changed.

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @test1(%arg0: i32, %arg1 : i32) -> i32 {
  // CHECK: dialect_conversion_bug_2
  %0 = "test.dialect_conversion_bug_1"(%arg0, %arg1) : (i32, i32) -> (i32)
  %1 = "test.dialect_conversion_bug_1"(%0, %arg1) : (i32, i32) -> (i32)
  func.return %1 : i32
}
