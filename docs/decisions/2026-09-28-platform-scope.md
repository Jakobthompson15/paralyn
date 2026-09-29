# Decision: Metal-only hardware for now; Linux replaces native Windows

Recorded 2026-09-28. The project owner decided both points.

## 1. Hardware: Metal only for now

- The only hardware available is an Apple M5. No NVIDIA, AMD or Windows machines are available, and none will be rented or acquired for now.
- Development continues on Metal.
- The NVIDIA (CUDA Driver/NVRTC) and AMD (HIP/ROCm) backends stay in the plan with the status **blocked: no hardware**. Their obligations are not removed.
- The CUDA backend code already written stays in the tree as `implemented_unqualified`. No NVIDIA or AMD qualification may be claimed.
- Anything that needs NVIDIA reference hardware is blocked by this decision. That includes PTX/SASS reference comparison and the first CUDA integrations for CuPy, Numba CUDA and CUDA Python.
- Revisit this decision if hardware becomes available, or if the owner authorizes cloud GPU rental.

## 2. Operating systems: Linux replaces native Windows

- The required operating systems are now **macOS and Linux**. Native Windows (and WSL) is **no longer required**.
- The Windows process runner, CMake settings and inventory script already in the tree are kept as unqualified, unbuilt code. No Windows support is claimed, and no further Windows work is scheduled.
- Linux obligations:
  - The runtime-only, compiler and CLI builds must compile and pass the CPU and host tests on Linux.
  - Linux packaging is required.
  - GPU execution on Linux arrives with the NVIDIA or AMD backends, which are blocked per decision 1.
  - Metal is macOS-only, so a Linux build without a vendor backend is the no-backend build. It must reject GPU execution honestly.
- Linux builds and tests can be checked in a local container (Docker on the Mac). That is build evidence only, never GPU evidence.
