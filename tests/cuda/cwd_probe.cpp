// TEST PROBE ONLY. A harmless shared library that the CUDA loader regression
// tests plant in a scratch working directory under the names of the CUDA
// driver/NVRTC libraries. If Paralyn ever loads it, its static initializer
// creates PARALYN_CWD_PROBE_LOADED in the current directory, which the tests
// treat as a failure (a library was loaded from the current working directory).
// It exports no CUDA symbols and is never installed.
#include <cstdio>

namespace {
struct Probe {
  Probe() {
    if (std::FILE *marker = std::fopen("PARALYN_CWD_PROBE_LOADED", "w")) {
      std::fputs("loaded from the current working directory\n", marker);
      std::fclose(marker);
    }
  }
} probe;
} // namespace

extern "C" int paralyn_cwd_probe_marker() { return 1; }
