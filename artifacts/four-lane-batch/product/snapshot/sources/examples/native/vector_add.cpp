#include <iostream>
#include <paralyn/native.hpp>
#include <vector>
using namespace paralyn::native;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("Usage: native_cpp MODULE.prk NEW_EVIDENCE_DIRECTORY");
    Context context;
    auto module = context.module(argv[1]);
    auto kernel = module.kernel("affine");
    auto queue = context.queue();
    const int n = 513;
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
      a[i] = float(i % 127 - 63) * 0.5f;
      b[i] = float(i % 37 - 18) * 0.25f;
    }
    auto ba = context.buffer(n * 4), bb = context.buffer(n * 4), bc = context.buffer(n * 4);
    // Errors retain the C ABI status and operation across the exception wrapper.
    bool rejected = false;
    try {
      (void)ba.view(n * 4, 4);
    } catch (const Error &e) {
      rejected = e.code == PR_OUT_OF_BOUNDS && e.operation == "view_create";
    }
    if (!rejected)
      throw std::runtime_error("C++ structured error propagation failed");
    auto moved = std::move(ba);
    if (ba.get() != 0 || moved.size() != n * 4)
      throw std::runtime_error("C++ move ownership failed");
    ba = std::move(moved);
    ba.write(a.data(), n * 4);
    bb.write(b.data(), n * 4);
    auto va = ba.view(0, n * 4, PR_READ), vb = bb.view(0, n * 4, PR_READ),
         vc = bc.view(0, n * 4, PR_WRITE);
    auto event = queue.launch(kernel, {3, 1, 1}, {256, 1, 1},
                              {Argument::buffer(va), Argument::buffer(vb), Argument::buffer(vc),
                               Argument::i32(n), Argument::f32(0.5f)});
    // All source/module handles may close before the consumer event is observed.
    va.close();
    vb.close();
    ba.close();
    bb.close();
    module.close();
    kernel.close();
    auto timing = event.wait();
    bc.read(out.data(), n * 4);
    if (!timing.completed ||
        !(timing.gpu_end_seconds > timing.gpu_start_seconds && timing.gpu_start_seconds > 0))
      throw std::runtime_error("Missing GPU timestamps");
    for (int i = 0; i < n; ++i)
      if (out[i] != a[i] * 0.5f + b[i])
        throw std::runtime_error("C++ CPU comparison mismatch");
    event.close();
    queue.close();
    vc.close();
    bc.close();
    context.evidence(argv[2]);
    context.close();
    std::cout << "Verification: PASS native C++ affine and RAII ownership\n";
    return 0;
  } catch (const Error &e) {
    std::cerr << e.operation << ": " << e.what() << " code=" << e.code << '\n';
    return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
