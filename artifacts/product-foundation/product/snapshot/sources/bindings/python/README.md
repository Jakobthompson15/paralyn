# Native arrays and local Python installation

The Python package and `paralyn/array.hpp` share the native C runtime. The array
operators are compiled from `operators.cu` into verified typed IR at build time.
Installed applications load that artifact and execute its generated Metal on a
physical GPU. They do not need LLVM, a CUDA toolkit, or the source checkout.

```python
import paralyn as p

with p.Context() as context:
    with p.asarray([1, 2, 3], context=context) as a:
        with p.asarray([4, 5, 6], context=context) as b:
            with p.affine(a, b, 2) as result:
                assert list(result.to_host()) == [6, 9, 12]
                print(result.wait())  # actual GPU completion and timestamps
```

`Context()` uses `PARALYN_DEVICE` when set, otherwise `auto`. An explicit selector,
including `Context("auto")`, takes precedence. Unsupported devices fail; no CPU
fallback exists. `add(a, b)` computes `a+b`; `affine(a, b, scale)` computes
`a*scale+b` with contraction disabled by the compiled operator policy.

## Supported contract

- Arrays own contiguous, one-dimensional FP32 storage. `dtype=p.float32` is the
  only dtype. `shape`, when supplied, must be exactly `(number_of_values,)`.
  Lengths range from zero through `INT32_MAX`, subject to device memory limits.
- `asarray` requires an explicit context and copies flat Python lists/tuples or a
  native-endian, one-dimensional, C-contiguous FP32 host buffer. Sequence values
  are explicitly rounded to binary32; bool, strings and overflowing finite
  values fail. FP64 buffers, raw bytes, nested sequences and strided buffers
  fail. NumPy is not required. There is no implicit layout or dtype conversion
  for buffer input.
- `to_host()` returns an independent `array.array('f')` copy. New operators
  allocate distinct results and require identical shapes and the same context.
  There is no broadcasting, slicing, in-place operation or implicit transfer
  between contexts.
- Nonempty operators enqueue work on the shared context's ordered GPU queue.
  Inputs may be closed after submission: the runtime retains submitted resources.
  `wait`, `to_host` and `close` observe completion failures. Uploaded arrays and
  empty results have no execution event; `wait()` returns `None` for them.
  Empty operators do not dispatch a fabricated GPU command.
- Explicitly closing a context prevents new work. Already submitted outputs
  remain readable through their retained native ownership. Array `close()` is
  idempotent. Context managers are recommended; an unhandled finalizer failure
  exits nonzero, matching the native API's failure policy.
- Operator-resource initialization is not promised to be safe for simultaneous
  calls from multiple host threads. Use one host thread per context in this version.

The C++ equivalent lives in `paralyn::arrays`: `Context`, move-only `Array`,
`asarray(Context&, const std::vector<float>&)`, `add`, `affine`, and
`Array::to_host()`. C++ `wait()` returns `std::optional<pr_event_info>`. See
`examples/native/arrays.cpp`. Both APIs execute the same bundled operator module.

## Build and install an offline wheel

After the normal CMake build has produced `paralyn`, `libparalyn_native.dylib`, and
`operators.prk`:

```sh
python3 bindings/python/build_wheel.py \
  --compiler build/paralyn --module build/operators.prk \
  --library build/libparalyn_native.dylib --output work/wheels
python3 -m pip install --no-index --no-deps work/wheels/paralyn-*.whl
python3 examples/native/arrays.py
```

The wheel builder currently supports macOS arm64 only. It reads the library's
actual minimum OS version for the platform tag and rejects unbundled native
dependencies. It recompiles and compares the operator artifact before packaging,
includes the library, operator source and IR, ABI metadata, hashes and licenses,
and uses deterministic ZIP metadata. The same input files produce identical wheel
bytes. This is a local packaging foundation, not a published package or a claim
of qualification across every Python or macOS version.

Package imports support Python 3.9 and later. Physical installed-package execution
has been checked with Python 3.9 and 3.14 on the current Apple M5/macOS 26 host.

The package discovers its bundled library and operators without environment
overrides. `PARALYN_LIBRARY` and `PARALYN_OPERATORS` are explicit developer
overrides. A source checkout can discover its `build/` outputs; that fallback is
enabled only when the matching checkout has `CMakeLists.txt`.

To repeat the isolated installation test, choose a **new directory outside the
source checkout**:

```sh
python3 bindings/python/test_install.py \
  --wheel work/wheels/paralyn-0.0.1-py3-none-macosx_26_0_arm64.whl \
  --output /tmp/paralyn-installed-test
```

The test creates a local virtual environment, installs offline, strips library
and module overrides, verifies that all runtime files resolve inside the installed
package, executes a copied example from the new directory, compares every value
against its own CPU reference, and audits both GPU events and retained source.
`--prepare-only` performs installation/import checks and explicitly reports that
it did not execute the GPU. Its timing covers this local workflow and does not
represent setup time on a fresh computer.

For development qualification:

```sh
PYTHONPATH=bindings/python python3 tests/native/test_arrays.py \
  --module build/operators.prk --artifacts artifacts/runs/arrays-new
build/array_tests build/operators.prk artifacts/runs/cpp-arrays-new
```

Examples write execution evidence only when `--artifacts` (Python), an output
directory argument (C++), or `PARALYN_ARTIFACT_DIR` is supplied. Arbitrary programs
choose their own explicit context evidence boundary.
