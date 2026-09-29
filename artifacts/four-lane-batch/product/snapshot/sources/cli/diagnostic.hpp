#pragma once
#include <stdexcept>
#include <string>
#include <utility>
namespace paralyn::cli {
// A user-facing failure with a stable identifier (see docs/terminal.md and
// docs/projects-and-cases.md) and the pipeline stage that produced it.
struct Diagnostic : std::runtime_error {
  std::string id, stage;
  Diagnostic(std::string identifier, std::string where, std::string text)
      : std::runtime_error(std::move(text)), id(std::move(identifier)), stage(std::move(where)) {}
};
} // namespace paralyn::cli
