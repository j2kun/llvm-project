//===- Pipeline.cpp - TableGen Pipeline Record Wrapper ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Pipeline class implementation.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Pipeline.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

Pipeline::Pipeline(const llvm::Record *def) : def(def) {
  assert(def->isSubClassOf("Pipeline") && 
         "must be subclass of 'Pipeline'");
}

llvm::StringRef Pipeline::getArgument() const {
  return def->getValueAsString("argument");
}

llvm::StringRef Pipeline::getOperation() const {
  return def->getValueAsString("operation");
}

llvm::StringRef Pipeline::getSummary() const {
  return def->getValueAsString("summary");
}

llvm::StringRef Pipeline::getDescription() const {
  return def->getValueAsString("description");
}

std::vector<PassOption> Pipeline::getOptions() const {
  auto *listInit = def->getValueAsListInit("options");
  std::vector<PassOption> options;
  for (const auto *optionInit : *listInit) {
    auto *optionDef = cast<DefInit>(optionInit)->getDef();
    options.emplace_back(optionDef);
  }
  return options;
}

const llvm::ListInit *Pipeline::getElements() const {
  return def->getValueAsListInit("elements");
}

//===----------------------------------------------------------------------===//
// PipelineElement
//===----------------------------------------------------------------------===//

PipelineElement::PipelineElement(const llvm::Init *init) {
  def = cast<DefInit>(init)->getDef();
}

bool PipelineElement::isPassElement() const {
  return def->isSubClassOf("PassElement");
}

bool PipelineElement::isNestedElement() const {
  return def->isSubClassOf("NestedElement");
}

bool PipelineElement::isPipelineElement() const {
  return def->isSubClassOf("PipelineRef");
}

llvm::StringRef PipelineElement::getPassName() const {
  assert(isPassElement() && "not a PassElement");
  return def->getValueAsString("pass");
}

llvm::StringRef PipelineElement::getOperation() const {
  assert(isNestedElement() && "not a NestedElement");
  return def->getValueAsString("operation");
}

llvm::StringRef PipelineElement::getPipelineName() const {
  assert(isPipelineElement() && "not a PipelineElement");
  return def->getValueAsString("pipeline");
}

const llvm::ListInit *PipelineElement::getElements() const {
  assert(isNestedElement() && "not a NestedElement");
  return def->getValueAsListInit("elements");
}

const llvm::ListInit *PipelineElement::getOptions() const {
  return def->getValueAsListInit("options");
}