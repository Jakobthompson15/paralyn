# PTX/SASS track: NVIDIA terms and fixture-provenance policy

Research briefing from 2026-09-28. **This is not legal advice.** The quotes were checked against the live sources listed. The risk ratings are the project's own reading and have not been reviewed by a lawyer.

## What NVIDIA's terms say

- The CUDA Toolkit EULA (v13.4, updated 2026-01-26), §1.2 "Limitations", says: *"You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform."* See <https://docs.nvidia.com/cuda/eula/index.html>.
  - The clause is in the online EULA from v11.5 (October 2021) onward. It is absent from v11.4. Press coverage in March 2024 connected it to CUDA 11.6+ installed files.
- §2.1 says: *"The SDK is licensed for you to develop applications only for use in systems with NVIDIA GPUs."* This binds anyone who accepted the SDK licence.
- The EULA does not define "output" or "SDK elements".
- The EULA is a contract. It binds people who downloaded, installed or used the SDK. Copyright and patent claims are separate risks.
- Laws relevant to reverse engineering for interoperability:
  - US: DMCA §1201(f).
  - EU: Directive 2009/24/EC Art. 6 and Art. 8. Art. 8 voids contract terms that conflict with Art. 6.
  - Bowers v. Baystate (Fed. Cir. 2003) is a US case where a contractual ban on reverse engineering was enforced.
- PTX ISA documentation carries NVIDIA's standard notice: no license is granted, and reproduction needs approval.
- NVIDIA publishes SASS only as opcode-name tables. It does not publish instruction encodings or semantics.

## Risk by input kind

| Input | Risk |
|---|---|
| User CUDA C++ source compiled by Paralyn's own Clang frontend (the current path) | Low. No SDK output is involved. SCALE, HIPIFY, chipStar and SYCLomatic take the same approach. |
| PTX from upstream LLVM/clang with `-nocudainc -nocudalib` and no NVIDIA headers, libdevice or ptxas | Low |
| Hand-written PTX based on the ISA documentation, with no examples copied | Low |
| PTX produced by nvcc/NVRTC | High for anyone bound by the EULA |
| SASS/cubin from ptxas/nvcc, and analysis with nvdisasm/cuobjdump | Highest |
| End users running Paralyn on their own nvcc-built binaries | Question for a lawyer |

## Project policy (adopted until a lawyer reviews it)

1. **PTX: go, clean-room only.** Test inputs may come only from:
   - upstream LLVM/clang with `-nocudainc -nocudalib` and Paralyn-owned headers;
   - hand-written PTX;
   - open generators that have been verified to run without NVIDIA components.

   Every test input records its tool, version, command line, and a statement that no NVIDIA SDK was present.
2. **Banned from the repository:** anything produced by nvcc, NVRTC, ptxas, nvdisasm or cuobjdump; cubins; NVIDIA headers; libdevice.
3. **Clean-room separation:** contributors who have accepted the CUDA EULA do not create PTX/SASS test inputs or study SDK output for this track.
4. **SASS translation is deferred** until a lawyer answers the questions below. No test inputs can be produced cleanly.
5. **Public description:** Paralyn is a compiler and runtime for CUDA *source* and open-toolchain PTX. It is not a "CUDA binary translation layer".

## Questions for a lawyer

1. Does "output generated using SDK elements" cover clang-produced PTX when an NVIDIA header or libdevice was used?
2. Can NVIDIA enforce §1.2 against maintainers who never accepted the EULA when end users run nvcc-built binaries through Paralyn?
3. Does §2.1 affect contributors who have the SDK installed?
4. How far does EU Art. 6/8 reach for EU contributors?
5. What patent exposure comes from implementing PTX/SASS semantics?
6. Can reimplemented CUDA API headers avoid copying NVIDIA's headers?
7. Is a runtime that *accepts* nvcc-produced PTX treated differently from the project itself producing or translating it?

## Precedents

- **ZLUDA** translates binary PTX. AMD withdrew the release in August 2024 because of AMD's own contract with the developer, not an NVIDIA action. It was rebuilt and in 2026 became a hobby project again.
- **SCALE** compiles CUDA source with a clang-based "nvcc" that has no NVIDIA dependency, and states it does not infringe NVIDIA's EULAs.
- **HIPIFY, SYCLomatic and chipStar** work at the source level.
- **CuPBoP** requires the CUDA Toolkit. Paralyn avoids that dependency.
