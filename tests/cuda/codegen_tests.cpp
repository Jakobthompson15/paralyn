// CPU-only tests of CUDA C++ generated from verified Paralyn IR. These compare
// generated text with reviewed golden files and check structural invariants.
// They do not compile with NVRTC or execute anything on a GPU.
#include "paralyn/artifact.hpp"
#include "paralyn/cuda_codegen.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace paralyn;
namespace {
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
void rejects(const std::function<void()> &f, const char *what) {
  bool rejected = false;
  try {
    f();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, std::string("accepted invalid input: ") + what);
}
bool contains(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }
std::string file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot read " + path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}
std::vector<unsigned char> bytes(const std::string &path) {
  auto text = file(path);
  return {text.begin(), text.end()};
}
Expr ref(const std::string &name, ScalarType type) { return {ExprKind::Ref, type, name, {}}; }
Expr builtin(const std::string &name) { return {ExprKind::Builtin, ScalarType::U32, name, {}}; }
Expr binary(const std::string &op, ScalarType type, Expr a, Expr b) {
  return {ExprKind::Binary, type, op, {std::move(a), std::move(b)}};
}
Expr load(const std::string &buffer, Expr index) {
  return {ExprKind::Load, ScalarType::F32, "", {ref(buffer, ScalarType::F32), std::move(index)}};
}
// Mirrors the frontend's verified IR for examples/native/kernels.cu and
// bindings/python/operators.cu (see `paralyn inspect`). The frontend-derived
// modules are compared against the same goldens when passed on the command line.
Kernel kernel(const std::string &name, ScalarType index_type, bool affine) {
  const auto global = binary("+", ScalarType::U32,
                             binary("*", ScalarType::U32, builtin("blockIdx.x"), builtin("blockDim.x")),
                             builtin("threadIdx.x"));
  Statement let{StmtKind::Let, "i", index_type,
                index_type == ScalarType::I32 ? Expr{ExprKind::Cast, ScalarType::I32, "", {global}} : global,
                {}, {}};
  const auto i = ref("i", index_type);
  Expr value = affine ? binary("+", ScalarType::F32,
                               binary("*", ScalarType::F32, load("a", i), ref("scale", ScalarType::F32)),
                               load("b", i))
                      : binary("+", ScalarType::F32, load("a", i), load("b", i));
  Statement store{StmtKind::Store, "", ScalarType::F32, value, load("out", i), {}};
  Statement guard{StmtKind::If, "", ScalarType::Bool,
                  binary("<", ScalarType::Bool, i, ref("n", index_type)), {}, {store}};
  Kernel k{name,
           {{"a", ScalarType::F32, true, true},
            {"b", ScalarType::F32, true, true},
            {"out", ScalarType::F32, true, false},
            {"n", index_type, false, false}},
           {let, guard}};
  if (affine) k.parameters.push_back({"scale", ScalarType::F32, false, false});
  return k;
}
void golden(const std::string &directory, const Kernel &k) {
  const auto expected = file(directory + "/" + k.name + ".cu");
  const auto actual = emit_cuda(k);
  if (actual != expected) {
    std::cerr << "--- expected " << k.name << "\n" << expected << "--- actual\n" << actual;
    throw std::runtime_error("generated CUDA differs from reviewed golden for " + k.name);
  }
}
void structure(const Kernel &k) {
  const auto source = emit_cuda(k);
  require(contains(source, "extern \"C\" __global__ void uc_kernel_" + k.name + "("),
          "unmangled, prefixed entrypoint missing");
  require(cuda_entrypoint(k) == "uc_kernel_" + k.name, "entrypoint helper disagrees");
  require(!contains(source, "__restrict__"), "aliasing views must not be declared restrict");
  require(!contains(source, "__fmaf") && !contains(source, "fmaf("), "explicit FMA emitted");
  require(!contains(source, "#include"), "generated device code must not depend on headers");
  // Every FP32 add/multiply uses the documented never-contracted intrinsics.
  std::size_t plus = 0, times = 0;
  for (std::size_t p = 0; (p = source.find("__fadd_rn(", p)) != std::string::npos; ++p) ++plus;
  for (std::size_t p = 0; (p = source.find("__fmul_rn(", p)) != std::string::npos; ++p) ++times;
  require(plus == 1, "expected exactly one FP32 add intrinsic");
  require(times == (k.parameters.size() == 5 ? 1u : 0u), "unexpected FP32 multiply intrinsic count");
  // Read-only IR buffers are const; the written buffer is not.
  require(contains(source, "const float *uc_arg_0") && contains(source, "const float *uc_arg_1") &&
              contains(source, "    float *uc_arg_2"),
          "buffer constness does not follow verified IR access");
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 2, "usage: cuda_codegen_tests GOLDEN_DIR [MODULE.prk ...]");
    const std::string directory = argv[1];
    const std::vector<Kernel> kernels = {kernel("vector_add", ScalarType::I32, false),
                                         kernel("affine", ScalarType::I32, true),
                                         kernel("array_add", ScalarType::U32, false),
                                         kernel("array_affine", ScalarType::U32, true)};
    for (const auto &k : kernels) {
      verify(k);
      golden(directory, k);
      structure(k);
    }

    // Numerical options: explicit, never fast math.
    const auto options = cuda_numerical_options();
    auto has = [&](const char *o) { return std::find(options.begin(), options.end(), o) != options.end(); };
    require(has("--fmad=false") && has("--ftz=false") && has("--prec-div=true") && has("--prec-sqrt=true"),
            "numerical policy options missing");
    for (const auto &o : options)
      require(!contains(o, "fast") && !contains(o, "gpu-architecture"),
              "fast math or implicit architecture in numerical options");

    // Literal encodings are exact and independent of NVRTC parsing.
    Kernel literals{"literals", {{"out", ScalarType::F32, true, false}, {"o", ScalarType::I32, true, false}}, {}};
    const auto zero = Expr{ExprKind::Literal, ScalarType::U32, "0", {}};
    literals.body.push_back({StmtKind::Store, "", ScalarType::F32, {ExprKind::Literal, ScalarType::F32, "0.1", {}},
                             load("out", zero), {}});
    literals.body.push_back({StmtKind::Store, "", ScalarType::I32,
                             {ExprKind::Literal, ScalarType::I32, "-2147483648", {}},
                             {ExprKind::Load, ScalarType::I32, "", {ref("o", ScalarType::I32), zero}}, {}});
    literals.body.push_back({StmtKind::Let, "octal", ScalarType::I32, {ExprKind::Literal, ScalarType::I32, "00012", {}}, {}, {}});
    literals.body.push_back({StmtKind::Let, "big", ScalarType::U32, {ExprKind::Literal, ScalarType::U32, "4294967295", {}}, {}, {}});
    literals.body.push_back({StmtKind::Let, "signed", ScalarType::I32,
                             {ExprKind::Cast, ScalarType::I32, "", {ref("big", ScalarType::U32)}}, {}, {}});
    const auto lit = emit_cuda(literals);
    float tenth = 0.1f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &tenth, 4);
    require(contains(lit, "__uint_as_float(" + std::to_string(bits) + "u)"), "float literal is not bit-exact");
    require(contains(lit, "static_cast<int>(-2147483648ll)"), "INT32_MIN literal is not well-formed");
    require(contains(lit, "static_cast<int>(12ll)"), "decimal literal became octal");
    require(contains(lit, "4294967295u"), "u32 literal lost its suffix");
    require(contains(lit, "static_cast<int>(uc_local_1)"), "explicit integer conversion missing");
    require(!contains(lit, "octal") && !contains(lit, " signed"), "source identifiers leaked into CUDA");

    // CUDA/C++ keywords as IR identifiers never reach generated source.
    auto keywords = kernels[0];
    keywords.name = "__global__";
    require(contains(emit_cuda(keywords), "void uc_kernel___global__("),
            "reserved-looking kernel name must be prefixed");
    keywords = kernels[0];
    keywords.name = "float";
    keywords.parameters[0].name = "int";
    keywords.body[1].body[0].expression.operands[0].operands[0].text = "int";
    const auto renamed = emit_cuda(keywords);
    require(contains(renamed, "uc_kernel_float(") && !contains(renamed, "*int") &&
                contains(renamed, "const float *uc_arg_0"),
            "keyword identifiers must be renamed");

    // Invalid IR is rejected before any CUDA text is produced.
    auto broken = kernels[0];
    broken.parameters[2].read_only = true; // store to read-only buffer
    rejects([&] { emit_cuda(broken); }, "store to read-only buffer");
    broken = kernels[0];
    broken.body[0].expression.operands[0].text = "undefined";
    rejects([&] { emit_cuda(broken); }, "undefined reference");
    broken = kernels[0];
    broken.body[1].expression.text = "/";
    rejects([&] { emit_cuda(broken); }, "unsupported operator");

    std::cout << "CUDA codegen goldens/structure for " << kernels.size() << " IR kernels: PASS\n";

    // Optional: modules produced by the real CUDA frontend (paralyn compile).
    std::size_t frontend = 0;
    for (int i = 2; i < argc; ++i) {
      const auto data = bytes(argv[i]);
      for (const auto &k : deserialize_module(data.data(), data.size())) {
        golden(directory, k);
        structure(k);
        ++frontend;
      }
    }
    if (argc > 2) {
      require(frontend == 4, "expected four frontend-derived add/affine kernels");
      std::cout << "Frontend-compiled modules match the same CUDA goldens (" << frontend << " kernels): PASS\n";
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "CUDA codegen test failed: " << e.what() << "\n";
    return 1;
  }
}
