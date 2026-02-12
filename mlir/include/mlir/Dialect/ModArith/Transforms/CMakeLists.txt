set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ModArith)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ModArith)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ModArith)
add_mlir_dialect_tablegen_target(MLIRModArithTransformsIncGen)

add_mlir_doc(Passes ModArithPasses ./ -gen-pass-doc)
