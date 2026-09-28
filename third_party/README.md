# Vendored CLI dependencies

These copies are build inputs; no network fetch occurs during configuration or runtime.

- CLI11 **2.4.2**, single header from <https://github.com/CLIUtils/CLI11/releases/download/v2.4.2/CLI11.hpp>, BSD-3-Clause, license in `cli11/LICENSE`.
- nlohmann/json **3.11.3**, single header from <https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp>, MIT, license in `nlohmann/LICENSE.MIT`.
- toml++ **3.4.0**, single header from <https://raw.githubusercontent.com/marzer/tomlplusplus/v3.4.0/toml.hpp> (tag commit `30172438cee64926dc41fdd9c11fb3ba5b2ba9de`), MIT, license in `tomlplusplus/LICENSE`. Git blob hashes of both files were checked against the upstream tag.

`SHA256SUMS.json` pins the actual checked-in bytes. All three are used only by the CLI. The native runtime and Python binding do not require any of them. Future FTXUI components are not yet imported.
