// RUN: mlir-opt -allow-unregistered-dialect -test-dialect-conversion-without-type-changes -verify-diagnostics %s | FileCheck %s

// Test that SSA values are properly replaced in dialect conversion even when
// types are not changed.

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @test1(%arg0: i32, %arg1 : i32) -> i32 {
  %0 = "test.bgv_mul"(%arg0, %arg1) : (i32, i32) -> (i64)
  %1 = "test.bgv_relin"(%0) : (i64) -> (i32)
  %2 = "test.bgv_sub"(%1, %arg0) : (i32, i32) -> (i32)
  %3 = "test.bgv_sub"(%2, %arg1) : (i32, i32) -> (i32)
  func.return %3 : i32
}
