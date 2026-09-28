# Vendored CLI dependencies

These copies are build inputs; no network fetch occurs during configuration or runtime.

- CLI11 **2.4.2**, single header from <https://github.com/CLIUtils/CLI11/releases/download/v2.4.2/CLI11.hpp>, BSD-3-Clause, license in `cli11/LICENSE`.
- nlohmann/json **3.11.3**, single header from <https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp>, MIT, license in `nlohmann/LICENSE.MIT`.

`SHA256SUMS.json` pins the actual checked-in bytes. Both are used only by the CLI. The native runtime and Python binding do not require either library. Future FTXUI and TOML components are not yet imported.
