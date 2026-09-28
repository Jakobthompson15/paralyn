#include "paralyn/executable.hpp"
#include "paralyn/artifact.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace paralyn;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void rejects(const std::function<void()> &f) {
  bool failed = false;
  try { f(); } catch (const std::runtime_error &) { failed = true; }
  require(failed, "invalid executable was accepted");
}
void put(std::vector<unsigned char> &b, std::uint32_t n) {
  for (unsigned i = 0; i < 32; i += 8) b.push_back(static_cast<unsigned char>(n >> i));
}
void str(std::vector<unsigned char> &b, const std::string &s) {
  put(b, static_cast<std::uint32_t>(s.size())); b.insert(b.end(), s.begin(), s.end());
}
void patch(std::vector<unsigned char> &b, std::size_t offset, std::uint32_t n) {
  for (unsigned i = 0; i < 32; i += 8) b.at(offset++) = static_cast<unsigned char>(n >> i);
}
ExecutableModule sample() {
  ExecutableModule m;
  m.producer = "test"; m.producer_version = "1"; m.source_name = "copy.metal";
  m.source = "#include <metal_stdlib>\nusing namespace metal;\n"
             "kernel void copy_values(device const float* a [[buffer(3)]], "
             "device float* out [[buffer(7)]], constant uint& n [[buffer(9)]], "
             "uint i [[thread_position_in_grid]]) { if(i<n) out[i]=a[i]; }\n";
  m.source_sha256 = source_sha256(m.source);
  m.entries = {{"copy_values", {
    {"a", ScalarType::F32, true, ResourceAccess::Read, 3, 4, 4},
    {"out", ScalarType::F32, true, ResourceAccess::ReadWrite, 7, 4, 4},
    {"n", ScalarType::U32, false, ResourceAccess::Read, 9, 4, 4}}, {64,1,1}}};
  return m;
}
} // namespace
int main() {
  try {
    require(source_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA-256");
    require(source_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc SHA-256");
    require(source_sha256(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "multi-block SHA-256");
    const auto m = sample();
    const auto bytes = serialize_executable(m);
    const auto loaded = deserialize_executable(bytes.data(), bytes.size());
    require(loaded.source == m.source && loaded.source_sha256 == m.source_sha256, "source provenance round trip");
    require(loaded.entries[0].parameters[1].binding == 7 && loaded.entries[0].required_block[0] == 64, "binding/block round trip");
    require(serialize_executable(loaded) == bytes, "serialization not deterministic");
    require(is_executable_artifact(bytes.data(), bytes.size()), "executable discriminator failed");
    require(!is_executable_artifact("PARALYN\0", 8), "legacy artifact discriminator changed");
    rejects([&] { deserialize_module(bytes.data(), bytes.size()); });
    for (std::size_t n = 0; n < bytes.size(); ++n)
      rejects([&] { deserialize_executable(bytes.data(), n); });
    rejects([&] { deserialize_executable(nullptr, 0); });
    rejects([&] { deserialize_executable(bytes.data(), executable_max_bytes + 1); });

    // Independent version-1 fixture fixes byte order and descriptor layout.
    std::vector<unsigned char> fixture{'P','A','R','A','L','Y','N','X'};
    put(fixture, 1); put(fixture, 1); str(fixture, m.target); put(fixture, 1);
    str(fixture, m.producer); str(fixture, m.producer_version); str(fixture, m.source_name);
    str(fixture, m.source_sha256); str(fixture, m.source);
    const auto entry_count = fixture.size(); put(fixture, 1); str(fixture, "copy_values");
    const auto block = fixture.size(); put(fixture, 64); put(fixture, 1); put(fixture, 1);
    const auto parameter_count = fixture.size(); put(fixture, 3);
    str(fixture, "a"); const auto parameter_type = fixture.size();
    put(fixture, 3); put(fixture, 1); put(fixture, 1); put(fixture, 3); put(fixture, 4); put(fixture, 4); put(fixture, 0);
    str(fixture, "out"); put(fixture, 3); put(fixture, 1); put(fixture, 3); put(fixture, 7); put(fixture, 4); put(fixture, 4); put(fixture, 0);
    str(fixture, "n"); put(fixture, 2); put(fixture, 0); put(fixture, 1); put(fixture, 9); put(fixture, 4); put(fixture, 4); put(fixture, 0);
    require(fixture == bytes, "container version-1 schema changed");
    for (auto offset : {std::size_t(8), std::size_t(12), std::size_t(16), entry_count,
                       block, parameter_count, parameter_type, parameter_type+4,
                       parameter_type+8, parameter_type+12, parameter_type+16}) {
      auto corrupt = bytes; patch(corrupt, offset, 0xffffffff);
      rejects([&] { deserialize_executable(corrupt.data(), corrupt.size()); });
    }
    auto corrupt = bytes; corrupt.push_back(0);
    rejects([&] { deserialize_executable(corrupt.data(), corrupt.size()); });
    auto check_bad = [&](const std::function<void(ExecutableModule&)> &f) {
      auto copy = m; f(copy); rejects([&] { serialize_executable(copy); });
    };
    check_bad([](auto &x) { x.entries.clear(); });
    check_bad([](auto &x) { x.entries.push_back(x.entries[0]); });
    check_bad([](auto &x) { x.entries[0].name = "bad-name"; });
    check_bad([](auto &x) { x.target = "cuda"; });
    check_bad([](auto &x) { x.numerical_policy = 2; });
    check_bad([](auto &x) { x.producer.clear(); });
    check_bad([](auto &x) { x.source_name = std::string(4097, 'a'); });
    check_bad([](auto &x) { x.source_name = "a\nb"; });
    check_bad([](auto &x) { x.source += " "; });
    check_bad([](auto &x) { x.entries[0].required_block = {1,0,1}; });
    check_bad([](auto &x) { x.entries[0].required_block = {1024,1024,1}; });
    check_bad([](auto &x) { x.entries[0].parameters[1].binding = 3; });
    check_bad([](auto &x) { x.entries[0].parameters[1].name = "a"; });
    check_bad([](auto &x) { x.entries[0].parameters[0].type = ScalarType::Bool; });
    check_bad([](auto &x) { x.entries[0].parameters[0].buffer = false; });
    check_bad([](auto &x) { x.entries[0].parameters[2].buffer = true; });
    check_bad([](auto &x) { x.entries[0].parameters[0].minimum_bytes = 0; });
    check_bad([](auto &x) { x.entries[0].parameters[0].alignment = 3; });
    check_bad([](auto &x) { x.entries[0].parameters[2].access = ResourceAccess::Write; });
    check_bad([](auto &x) { x.entries[0].parameters[2].minimum_bytes = 8; });
    for (const auto &suffix : {"\n#pragma clang fp contract(fast)\n", "\n#include <other>\n",
                               "\n#define X 1\n", "\n_Pragma(\"x\")\n", "/* unterminated",
                               "\n%:pragma clang fp contract(fast)\n", "\n?" "?=pragma once\n"})
      check_bad([&](auto &x) { x.source += suffix; x.source_sha256 = source_sha256(x.source); });
    check_bad([](auto &x) { x.source += '\0'; x.source_sha256 = source_sha256(x.source); });
    check_bad([](auto &x) { x.source = std::string(executable_max_source_bytes+1, ' '); });
    const auto replace = [](std::string &source, const std::string &before, const std::string &after) {
      const auto offset = source.find(before);
      require(offset != std::string::npos, "test source replacement absent");
      source.replace(offset,before.size(),after);
    };
    for (const auto &mutation : std::vector<std::pair<std::string,std::string>>{
      {"device const float* a", "constant float& a"},
      {"device const float* a", "device float* a"},
      {"device const float* a", "device const F* a"},
      {"device const float* a", "device const float& a"},
      {"device const float* a", "device const const float* a"},
      {"constant uint& n", "constant uint* n"},
      {"buffer(3)", "buffer(1+2)"},
      {"thread_position_in_grid", "threadgroup(0)"},
      {"kernel void copy_values", "kernel void unlisted"},
      {"uint i [[thread_position_in_grid]]) {", "uint i [[thread_position_in_grid]]); void unused() {"},
    }) check_bad([&](auto &x) {
      replace(x.source,mutation.first,mutation.second); x.source_sha256=source_sha256(x.source);
    });
    check_bad([](auto &x) {
      x.source += "\nkernel void duplicate() {}\n"; x.source_sha256=source_sha256(x.source);
    });
    check_bad([](auto &x) {
      x.source += x.source.substr(x.source.find("kernel")); x.source_sha256=source_sha256(x.source);
    });
    check_bad([](auto &x) {
      x.source += "\nconstant auto s=R\"tag(text \" kernel void spoof() {} )tag\";";
      x.source_sha256=source_sha256(x.source);
    });
    auto qualifier = m;
    replace(qualifier.source,"device const float* a","const device float* a");
    qualifier.source_sha256=source_sha256(qualifier.source); verify_executable(qualifier);
    replace(qualifier.source,"const device float* a","device float const* a");
    qualifier.source_sha256=source_sha256(qualifier.source); verify_executable(qualifier);
    for (const std::string invalid : {"\x80", "\xff", "\xc0\xaf", "\xe0\x80\x80",
                                      "\xed\xa0\x80", "\xf0\x80\x80\x80", "\xf4\x90\x80\x80",
                                      "\xe2\x82", "\xf5\x80\x80\x80", "\xc2 "}) {
      check_bad([&](auto &x) { x.source += "\n// " + invalid; x.source_sha256 = source_sha256(x.source); });
      check_bad([&](auto &x) { x.source_name = invalid; });
    }
    auto comment = m; comment.source += "\n/* #pragma ignored comment */\n// #include <ignored>\n";
    comment.source_sha256 = source_sha256(comment.source); verify_executable(comment);
    comment.source += "// caf\xc3\xa9 \xe9\x9b\xaa \xf0\x9f\x8e\xaf\n";
    comment.source_name = "caf\xc3\xa9.metal";
    comment.source_sha256 = source_sha256(comment.source); verify_executable(comment);
    std::cout << "Executable artifact schema, provenance, source profile and limits: PASS\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
