#include <paralyn/array.hpp>
#include <iostream>
int main(int argc, char **argv) {
  try {
    if (argc != 1 && argc != 2)
      throw std::runtime_error("Usage: native_arrays [NEW_EVIDENCE_DIRECTORY]");
    paralyn::arrays::Context context;
    std::vector<float> a(1003), b(1003);
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = float(int(i % 31) - 15) * 0.5f;
      b[i] = float(int(i % 17) - 8) * 0.25f;
    }
    auto da = paralyn::arrays::asarray(context, a);
    auto db = paralyn::arrays::asarray(context, b);
    auto sum = paralyn::arrays::add(da, db);
    auto result = paralyn::arrays::affine(sum, db, 0.5f);
    auto actual = result.to_host();
    for (std::size_t i = 0; i < a.size(); ++i)
      if (actual[i] != (a[i] + b[i]) * 0.5f + b[i])
        throw std::runtime_error("Native C++ array CPU reference mismatch");
    auto timing = result.wait();
    if (!timing || !timing->completed || !(0 < timing->gpu_start_seconds &&
                                         timing->gpu_start_seconds < timing->gpu_end_seconds))
      throw std::runtime_error("Missing actual GPU completion timestamps");
    std::cout << "Device: " << context.device().name << "\n";
    da.close(); db.close(); sum.close(); result.close();
    const char *environment = std::getenv("PARALYN_ARTIFACT_DIR");
    if (argc == 2) context.evidence(argv[1]);
    else if (environment && *environment) context.evidence(environment);
    context.close();
    std::cout << "Verification: PASS native C++ arrays (1003 values, add + affine)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Native arrays failed: " << error.what() << '\n';
    return 1;
  }
}
