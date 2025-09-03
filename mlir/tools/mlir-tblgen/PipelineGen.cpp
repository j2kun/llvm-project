//===- PipelineGen.cpp - MLIR pipeline registration generator -------------===//
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

#include "PipelineGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory pipelineGenCat("Options for -gen-pipeline-*");
static llvm::cl::opt<std::string>
    groupName("pipelineGroupName", llvm::cl::desc("The name of this group of pipelines"),
              llvm::cl::cat(pipelineGenCat));

namespace {
/// Wrapper class around a pipeline definition record.
class PipelineDef {
public:
  explicit PipelineDef(const llvm::Record *def) : def(def) {}

  /// Get the command line argument for this pipeline.
  StringRef getArgument() const { return def->getValueAsString("argument"); }

  /// Get the target operation for this pipeline.
  StringRef getOperation() const { return def->getValueAsString("operation"); }

  /// Get the summary for this pipeline.
  StringRef getSummary() const { return def->getValueAsString("summary"); }

  /// Get the description for this pipeline.
  StringRef getDescription() const {
    return def->getValueAsString("description");
  }

  /// Get the pipeline options.
  std::vector<PassOption> getOptions() const {
    auto *listInit = def->getValueAsListInit("options");
    std::vector<PassOption> options;
    for (const auto *optionInit : *listInit) {
      auto *optionDef = llvm::cast<llvm::DefInit>(optionInit)->getDef();
      options.emplace_back(optionDef);
    }
    return options;
  }

  /// Get the pipeline elements.
  const llvm::ListInit *getElements() const {
    return def->getValueAsListInit("elements");
  }

  /// Get the underlying record.
  const llvm::Record *getDef() const { return def; }

private:
  const llvm::Record *def;
};

/// Wrapper class around a pipeline element record.
class PipelineElement {
public:
  explicit PipelineElement(const llvm::Init *init) {
    def = llvm::cast<llvm::DefInit>(init)->getDef();
  }

  /// Check if this is a pass element.
  bool isPassElement() const {
    return def->isSubClassOf("PassElement");
  }

  /// Check if this is a nested element.
  bool isNestedElement() const {
    return def->isSubClassOf("NestedElement");
  }

  /// Check if this is a pipeline reference element.
  bool isPipelineElement() const {
    return def->isSubClassOf("PipelineRef");
  }

  /// Get the pass name (for PassElement).
  StringRef getPassName() const {
    return def->getValueAsString("pass");
  }

  /// Get the target operation (for NestedElement).
  StringRef getOperation() const {
    return def->getValueAsString("operation");
  }

  /// Get the pipeline name (for PipelineElement).
  StringRef getPipelineName() const {
    return def->getValueAsString("pipeline");
  }

  /// Get the nested elements (for NestedElement).
  const llvm::ListInit *getElements() const {
    return def->getValueAsListInit("elements");
  }

  /// Get the options as a list of strings.
  const llvm::ListInit *getOptions() const {
    return def->getValueAsListInit("options");
  }

private:
  const llvm::Record *def;
};
} // namespace

/// Extract the list of pipelines from the TableGen records.
static std::vector<PipelineDef> getPipelines(const RecordKeeper &records) {
  std::vector<PipelineDef> pipelines;

  for (const auto *def : records.getAllDerivedDefinitions("Pipeline"))
    pipelines.emplace_back(def);

  return pipelines;
}

const char *const pipelineHeader = R"(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//
)";

/// Generate the options struct for a pipeline.
static void emitPipelineOptionsStruct(const PipelineDef &pipeline,
                                      llvm::raw_ostream &os) {
  StringRef pipelineName = pipeline.getDef()->getName();
  auto options = pipeline.getOptions();

  // Emit the struct only if the pipeline has at least one option.
  if (options.empty())
    return;

  os << formatv("struct {0}Options {{\n", pipelineName);

  for (const PassOption &opt : options) {
    std::string type = opt.getType().str();

    if (opt.isListOption())
      type = "::llvm::SmallVector<" + type + ">";

    os.indent(2) << formatv("{0} {1}", type, opt.getCppVariableName());

    if (std::optional<StringRef> defaultVal = opt.getDefaultValue())
      os << " = " << defaultVal;

    os << ";\n";
  }

  os << "};\n\n";
}

/// Generate code for a single pipeline element.
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

/// Generate the builder function for a pipeline.
static void emitPipelineBuilder(const PipelineDef &pipeline,
                                llvm::raw_ostream &os) {
  StringRef pipelineName = pipeline.getDef()->getName();
  auto options = pipeline.getOptions();

  os << "void build"
     << llvm::convertToCamelFromSnakeCase(pipelineName, true)
     << "Pipeline(::mlir::OpPassManager &pm";

  if (!options.empty()) {
    os << ", const " << pipelineName << "Options &options";
  }

  os << ") {\n";

  // Process pipeline elements
  const llvm::ListInit *elements = pipeline.getElements();
  for (const auto *elemInit : *elements) {
    PipelineElement element(elemInit);
    emitPipelineElement(element, pipeline, os);
  }

  os << "}\n\n";
}

/// Generate the registration code for all pipelines.
static void emitPipelineRegistrations(llvm::ArrayRef<PipelineDef> pipelines,
                                      llvm::raw_ostream &os,
                                      StringRef groupName) {
  os << "void register" << groupName << "Pipelines() {\n";

  for (const auto &pipeline : pipelines) {
    StringRef pipelineName = pipeline.getDef()->getName();
    StringRef argument = pipeline.getArgument();
    StringRef summary = pipeline.getSummary();
    auto options = pipeline.getOptions();

    if (!options.empty()) {
      os << "  ::mlir::PassPipelineRegistration<" << pipelineName << "Options>(\n";
    } else {
      os << "  ::mlir::PassPipelineRegistration<>(\n";
    }

    os << "    \"" << argument << "\",\n";
    os << "    \"" << summary << "\",\n";
    os << "    build"
       << llvm::convertToCamelFromSnakeCase(pipelineName, true)
       << "Pipeline);\n\n";
  }

  os << "}\n\n";
}

void mlir::emitPipelineDecls(const RecordKeeper &recordKeeper,
                             llvm::raw_ostream &os, StringRef groupName) {
  std::vector<PipelineDef> pipelines = getPipelines(recordKeeper);
  if (pipelines.empty())
    return;

  os << formatv(pipelineHeader, groupName + " Pipeline Declarations");

  os << "namespace mlir {\n\n";

  // Emit pipeline options structs
  for (const auto &pipeline : pipelines) {
    emitPipelineOptionsStruct(pipeline, os);
  }

  // Emit pipeline builder function declarations
  for (const auto &pipeline : pipelines) {
    StringRef pipelineName = pipeline.getDef()->getName();
    auto options = pipeline.getOptions();

    os << "void build"
       << llvm::convertToCamelFromSnakeCase(pipelineName, true)
       << "Pipeline(::mlir::OpPassManager &pm";

    if (!options.empty()) {
      os << ", const " << pipelineName << "Options &options";
    }

    os << ");\n";
  }

  os << "\n";

  // Emit registration function declaration
  os << "void register" << groupName << "Pipelines();\n\n";

  os << "} // namespace mlir\n";
}

void mlir::emitPipelineImpls(const RecordKeeper &recordKeeper,
                             llvm::raw_ostream &os, StringRef groupName) {
  std::vector<PipelineDef> pipelines = getPipelines(recordKeeper);
  if (pipelines.empty())
    return;

  os << formatv(pipelineHeader, groupName + " Pipeline Implementations");

  os << "namespace mlir {\n\n";

  // Emit pipeline builder functions
  for (const auto &pipeline : pipelines) {
    emitPipelineBuilder(pipeline, os);
  }

  // Emit registration function
  emitPipelineRegistrations(pipelines, os, groupName);

  os << "} // namespace mlir\n";
}

void mlir::emitPipelineDoc(const RecordKeeper &recordKeeper,
                           llvm::raw_ostream &os) {
  std::vector<PipelineDef> pipelines = getPipelines(recordKeeper);
  if (pipelines.empty())
    return;

  os << "# Generated Pipeline Documentation\n\n";

  for (const auto &pipeline : pipelines) {
    os << "## " << pipeline.getArgument() << "\n\n";

    if (!pipeline.getSummary().empty()) {
      os << pipeline.getSummary() << "\n\n";
    }

    if (!pipeline.getDescription().empty()) {
      os << pipeline.getDescription() << "\n\n";
    }

    auto options = pipeline.getOptions();
    if (!options.empty()) {
      os << "### Options\n\n";
      for (const auto &option : options) {
        os << "- `--" << option.getArgument() << "`: "
           << option.getDescription() << "\n";
      }
      os << "\n";
    }
  }
}

// Register the generators
static mlir::GenRegistration genPipelineDecls(
    "gen-pipeline-decls", "Generate pipeline declarations",
    [](const RecordKeeper &records, llvm::raw_ostream &os) {
      mlir::emitPipelineDecls(records, os, groupName);
      return false;
    });

static mlir::GenRegistration genPipelineImpls(
    "gen-pipeline-impls", "Generate pipeline implementations",
    [](const RecordKeeper &records, llvm::raw_ostream &os) {
      mlir::emitPipelineImpls(records, os, groupName);
      return false;
    });

static mlir::GenRegistration genPipelineDoc(
    "gen-pipeline-doc", "Generate pipeline documentation",
    [](const RecordKeeper &records, llvm::raw_ostream &os) {
      mlir::emitPipelineDoc(records, os);
      return false;
    });
