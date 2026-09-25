#include "unicuda/frontend.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
const std::string program = R"CU(
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int grid_value() { return 4; }
int block_value() { return 64; }
int count_value() { return 251; }
int main() {
  std::vector<float> values(251, 1.0f);
  float *a, *b, *c;
  cudaMalloc(&a, values.size() * sizeof(float));
  cudaMalloc(&b, values.size() * sizeof(float));
  cudaMalloc(&c, values.size() * sizeof(float));
  cudaMemcpy(a, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
  add<<<grid_value(), block_value()>>>(a, b, c, count_value());
  cudaDeviceSynchronize();
  std::puts("host main preserved");
  cudaFree(a); cudaFree(b); cudaFree(c);
  return 0;
}
)CU";

struct TemporarySources {
  std::filesystem::path root = std::filesystem::temp_directory_path() /
                               ("unicuda_frontend_tests_" + std::to_string(getpid()));
  TemporarySources() { std::filesystem::create_directories(root); }
  ~TemporarySources() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  std::string write(const std::string &name, const std::string &source) {
    auto path = root / name;
    std::ofstream out(path);
    out << source;
    if (!out)
      throw std::runtime_error("cannot write test source");
    return path.string();
  }
};

void rejects(TemporarySources &files, const std::string &source, const std::string &feature,
             const std::string &filename) {
  const auto path = files.write(filename, source);
  try {
    (void)unicuda::compile_source(path);
  } catch (const std::exception &error) {
    const std::string message = error.what();
    require(message.find(feature) != std::string::npos,
            "wrong rejection for " + filename + ": " + message);
    require(message.find(files.root.string()) != std::string::npos &&
                (message.find(".cu:") != std::string::npos ||
                 message.find(".hpp:") != std::string::npos),
            "missing source location for " + filename + ": " + message);
    return;
  }
  throw std::runtime_error("unexpected acceptance: " + filename);
}
} // namespace

int main() {
  try {
    TemporarySources files;
    const auto result = unicuda::compile_source(files.write("ordinary.cu", program));
    require(result.kernels.size() == 1 && result.launches.size() == 1, "kernel/launch extraction");
    const auto &kernel = result.kernels.front();
    require(kernel.parameters.size() == 4 && kernel.parameters[0].read_only &&
                !kernel.parameters[2].read_only,
            "typed buffer parameters");
    require(kernel.body.size() == 2 && kernel.body[0].kind == unicuda::StmtKind::Let &&
                kernel.body[1].kind == unicuda::StmtKind::If,
            "structured statements");
    require(kernel.body[0].expression.kind == unicuda::ExprKind::Cast &&
                kernel.body[0].expression.type == unicuda::ScalarType::I32 &&
                kernel.body[0].expression.operands[0].type == unicuda::ScalarType::U32,
            "CUDA unsigned indexing conversion must remain explicit");
    const auto &store = kernel.body[1].body.at(0);
    require(store.kind == unicuda::StmtKind::Store && store.target.operands[0].text == "c" &&
                store.expression.operands[0].kind == unicuda::ExprKind::Load,
            "typed loads and stores");
    require(store.expression.line != 0 && store.expression.column != 0, "expression source span");
    const auto &host = result.rewritten_host;
    require(host.find("std::vector<float> values(251, 1.0f)") != std::string::npos &&
                host.find("std::puts(\"host main preserved\")") != std::string::npos,
            "ordinary host code must survive");
    require(host.find("<<<") == std::string::npos && host.find("__global__") == std::string::npos,
            "CUDA-only syntax remains in rewritten host");
    const auto grid = host.find("__unicuda_generated_grid = (grid_value())");
    const auto block = host.find("__unicuda_generated_block = (block_value())");
    const auto check = host.find("validate_launch_configuration");
    const auto arg = host.find("__unicuda_generated_argument_3 = (count_value())");
    require(grid != std::string::npos && grid < block && block < check && check < arg,
            "configuration must be evaluated before kernel arguments");
    require(host.find("count_value()", arg + 54) == std::string::npos,
            "kernel argument evaluated more than once");

    rejects(files, "__global__ void bad(float* a) { __shared__ float tile[32]; a[0] = tile[0]; }",
            "shared_memory", "shared.cu");
    rejects(files, "__global__ void bad(float* a) { a[0] = a[0] - a[1]; }", "operator_-",
            "operator.cu");
    rejects(files, "__global__ void bad(double* a) { a[0] = a[1]; }", "device_type_double",
            "double.cu");
    rejects(files,
            "#if __CUDA_ARCH__\n__global__ void hidden(float* a) { a[0]=a[1]; }\n#endif\n" +
                program,
            "device_architecture_preprocessing", "arch.cu");
    rejects(files,
            "__global__ void k(float* a) { a[0]=a[1]; }\n#define CALL(a) k<<<1,1>>>(a)\n"
            "int main(){ float* a=nullptr; CALL(a); }",
            "macro_generated_launch", "macro.cu");
    rejects(files,
            "__global__ void k(float* a) { a[0]=a[1]; }\n"
            "int main(){ float* a=nullptr; k<<<1,1,64>>>(a); }",
            "dynamic_shared_memory", "shared_launch.cu");
    rejects(files,
            "__global__ void k(float* a) { a[0]=a[1]; }\n"
            "int main(){ float* a=nullptr; k<<<1,1,0,(cudaStream_t)1>>>(a); }",
            "non_default_stream", "stream_launch.cu");
    files.write("conditional.hpp", "#if defined(__CUDA_ARCH__)\n#endif\n");
    rejects(files, "#include \"conditional.hpp\"\n" + program, "device_architecture_preprocessing",
            "header_arch.cu");
    rejects(files, "#if defined(__CUDACC__)\n#endif\n" + program, "cuda_conditioned_preprocessing",
            "cudacc.cu");
    rejects(files, "int source_line = __LINE__;\n" + program,
            "source_location_or_context_expression", "location.cu");
    std::cout << "Frontend tests: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Frontend tests: FAIL: " << error.what() << '\n';
    return 1;
  }
}
