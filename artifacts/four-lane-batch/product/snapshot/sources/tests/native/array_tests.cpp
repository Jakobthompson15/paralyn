#include <paralyn/array.hpp>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace paralyn::arrays;
namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()> &call) {
  bool rejected = false;
  try { call(); } catch (const std::exception &) { rejected = true; }
  require(rejected, "invalid array operation was accepted");
}
}
int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: array_tests MODULE.prk NEW_EVIDENCE_DIRECTORY");
    Context context("", argv[1]);
    unsigned commands = 0;
    for (std::size_t n : {0, 1, 17, 256, 257, 1003, 1000003}) {
      std::vector<float> a(n), b(n), expected(n);
      for (std::size_t i = 0; i < n; ++i) {
        a[i] = float(int(i % 61) - 30) * 0.5f;
        b[i] = float(int(i % 37) - 18) * 0.25f;
        expected[i] = (a[i] + b[i]) * 0.5f + (a[i] * -2.0f + b[i]);
      }
      auto da = asarray(context, a), db = asarray(context, b);
      require(da.size() == n && da.shape()[0] == n && da.nbytes() == n * 4, "bad array metadata");
      require(!da.wait(), "upload fabricated a GPU event");
      if (n) {
        a[0] = 9999;
        require(da.to_host()[0] != 9999, "host input was borrowed");
      }
      auto sum = add(da, db), scaled = affine(da, db, -2.0f);
      auto result = affine(sum, scaled, 0.5f);
      da.close(); db.close();
      for (const auto *array : {&sum, &scaled, &result}) {
        auto event = array->wait();
        require(bool(event) == bool(n), "empty operation or nonempty event contract failed");
        if (event) {
          require(event->completed && 0 < event->gpu_start_seconds &&
                      event->gpu_start_seconds < event->gpu_end_seconds, "missing real GPU timestamps");
          ++commands;
        }
      }
      sum.close(); scaled.close();
      require(result.to_host() == expected, "independent CPU array comparison failed");
      auto moved = std::move(result);
      rejects([&] { result.to_host(); });
      require(moved.to_host() == expected, "move lost array ownership");
      moved.close(); moved.close();
      rejects([&] { moved.to_host(); });
    }
    {
      auto a = asarray(context, {1, 2, 3});
      auto alias = add(a, a);
      require(alias.to_host() == std::vector<float>({2, 4, 6}), "same-input alias failed");
      ++commands;
      auto short_array = asarray(context, {1});
      rejects([&] { add(a, short_array); });
      Context other("", argv[1]);
      auto foreign = asarray(other, {1, 2, 3});
      rejects([&] { add(a, foreign); });
      Context unavailable("", "/nonexistent/paralyn/operators.prk");
      auto input = asarray(unavailable, {1});
      rejects([&] { add(input, input); });
    }
    require(commands == 19, "unexpected GPU command count");
    context.evidence(argv[2]);
    context.close();
    rejects([&] { asarray(context, {1}); });
    // Both explicit context close and ordinary context-wrapper destruction must
    // preserve a submitted result. The retained Array owns the shared state.
    std::ofstream lifetime_evidence(std::filesystem::path(argv[2]) / "lifetimes.json");
    lifetime_evidence.exceptions(std::ios::badbit | std::ios::failbit);
    lifetime_evidence << std::setprecision(17) << "{\"events\":[";
    for (bool close_parent : {true, false}) {
      std::unique_ptr<Array> retained;
      {
        Context parent("", argv[1]);
        auto a = asarray(parent, {1, 2, 3});
        auto b = asarray(parent, {4, 5, 6});
        retained = std::make_unique<Array>(add(a, b));
        a.close();
        b.close();
        if (close_parent) {
          parent.close();
          rejects([&] { asarray(parent, {1}); });
        }
      }
      require(retained->to_host() == std::vector<float>({5, 7, 9}),
              "parent lifetime invalidated the submitted output");
      const auto event = retained->wait();
      require(event && event->completed && 0 < event->gpu_start_seconds &&
                  event->gpu_start_seconds < event->gpu_end_seconds,
              "retained output lacks actual GPU completion");
      lifetime_evidence << (close_parent ? "" : ",") << "{\"parent\":\""
                        << (close_parent ? "explicitly_closed" : "wrapper_destroyed")
                        << "\",\"gpu_start_seconds\":" << event->gpu_start_seconds
                        << ",\"gpu_end_seconds\":" << event->gpu_end_seconds << '}';
      retained->close();
    }
    lifetime_evidence << "],\"status\":\"verified\",\"cpu_fallback\":false}\n";
    lifetime_evidence.close();
    std::cout << "Verification: PASS native C++ arrays (19 source-linked + 2 retained-output GPU events, independent CPU references)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "C++ array qualification failed: " << error.what() << '\n';
    return 1;
  }
}
