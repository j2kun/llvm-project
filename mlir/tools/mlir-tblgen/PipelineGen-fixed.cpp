// Simplified version of emitPipelineElement function
static void emitPipelineElement(const PipelineElement &element,
                                const PipelineDef &pipeline,
                                llvm::raw_ostream &os,
                                int indent = 2) {
  std::string indentStr(indent, ' ');

  if (element.isPassElement()) {
    StringRef passName = element.getPassName();
    os << indentStr << "// Add pass: " << passName << "\n";
    
    // For now, just generate simple passes without options
    // TODO: Add option handling support for list<string> options
    os << indentStr << "pm.addPass(mlir::create" 
       << llvm::convertToCamelFromSnakeCase(passName, true) 
       << "Pass());\n";
       
  } else if (element.isNestedElement()) {
    StringRef opType = element.getOperation();
    os << indentStr << "// Nested pipeline for " << opType << "\n";
    os << indentStr << "{\n";
    os << indentStr << "  auto nestedPM = pm.nest<" << opType << ">();\n";
    
    // Process nested elements
    const llvm::ListInit *elements = element.getElements();
    for (const auto *elemInit : *elements) {
      PipelineElement nestedElem(elemInit);
      emitPipelineElement(nestedElem, pipeline, os, indent + 2);
    }
    
    os << indentStr << "}\n";
    
  } else if (element.isPipelineElement()) {
    StringRef pipelineName = element.getPipelineName();
    os << indentStr << "// Add pipeline: " << pipelineName << "\n";
    
    // For now, just generate simple pipeline calls
    // TODO: Add option handling support
    std::string emptyOptionsStruct = pipelineName.str() + "Options{}";
    os << indentStr << "build" 
       << llvm::convertToCamelFromSnakeCase(pipelineName, true)
       << "Pipeline(pm, " << emptyOptionsStruct << ");\n";
  }
  
  os << "\n";
}