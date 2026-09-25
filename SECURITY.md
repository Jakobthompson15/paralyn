# Security policy

Paralyn is an experimental local compiler/runtime. No stable release or security support window is promised yet. Actual tested versions and limitations are recorded in `docs/status.md`.

## Trust boundary

`paralyn run` compiles and executes the supplied native host C++ program with the caller's privileges. It is not a sandbox for untrusted CUDA source. Host programs can perform normal native process/file/network operations. GPU code is compiled by the platform toolchain and runs through public Metal APIs.

The compatibility layer validates allocation tokens, byte bounds at API boundaries, launch geometry, and supported IR. Those checks do not prove arbitrary device memory accesses safe. Unsupported language diagnostics are a compatibility boundary, not a general adversarial-code verifier. There is no remote execution service or distributed daemon in Gate A.

## Reporting

Use [GitHub private vulnerability reporting](https://github.com/Jakobthompson15/paralyn/security/advisories/new) for suspected security bugs. Do not submit sensitive exploit details in public issues. If that form is unavailable, open an issue requesting a private security contact without including exploit details, secrets, or personal data.

Include the Paralyn revision, compiler/OS/GPU versions, a minimized reproducer, observed behavior, expected boundary, and relevant diagnostics. Remove secrets and personal data from logs. Maintainers should acknowledge the report, reproduce it where possible, agree on disclosure timing, and document any fix and affected versions. No response-time commitment is made until maintainers establish one.

Compiler crashes, unchecked host-side buffer copies, stale-token confusion, command execution through malformed CLI arguments, and unexpected resource lifetime violations are useful reports. Ordinary native execution explicitly requested by the input host program is expected behavior.
