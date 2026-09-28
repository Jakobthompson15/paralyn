#include "paralyn/executable.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace paralyn {
namespace {
constexpr unsigned char magic[] = {'P', 'A', 'R', 'A', 'L', 'Y', 'N', 'X'};
constexpr std::uint32_t version = 1, max_text = 4096, max_entries = 64, max_parameters = 31;
[[noreturn]] void bad(const std::string &message) {
  throw std::runtime_error("ParalynError: invalid executable artifact: " + message);
}
bool identifier(const std::string &s) {
  const auto letter = [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
  };
  return !s.empty() && s.size() <= 255 && letter(s.front()) &&
         std::all_of(s.begin(), s.end(),
                     [&](unsigned char c) { return letter(c) || (c >= '0' && c <= '9'); });
}
bool utf8(const std::string &s) {
  for (std::size_t i = 0; i < s.size();) {
    const auto first = static_cast<unsigned char>(s[i++]);
    if (first < 0x80)
      continue;
    unsigned count;
    std::uint32_t value, minimum;
    if (first >= 0xc2 && first <= 0xdf) {
      count = 1;
      value = first & 0x1f;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      count = 2;
      value = first & 0x0f;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      count = 3;
      value = first & 0x07;
      minimum = 0x10000;
    } else
      return false;
    if (count > s.size() - i)
      return false;
    while (count--) {
      const auto next = static_cast<unsigned char>(s[i++]);
      if ((next & 0xc0) != 0x80)
        return false;
      value = (value << 6) | (next & 0x3f);
    }
    if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
      return false;
  }
  return true;
}
void text(const std::string &s, bool required = true) {
  if ((required && s.empty()) || s.size() > max_text || !utf8(s) ||
      std::any_of(s.begin(), s.end(), [](unsigned char c) { return c < 32 || c == 127; }))
    bad("metadata must be bounded, nonempty UTF-8 without control bytes");
}
std::uint32_t scalar_code(ScalarType t) {
  switch (t) {
  case ScalarType::I32:
    return 1;
  case ScalarType::U32:
    return 2;
  case ScalarType::F32:
    return 3;
  default:
    bad("only i32, u32 and f32 resource types are supported");
  }
}
ScalarType scalar_type(std::uint32_t value) {
  switch (value) {
  case 1:
    return ScalarType::I32;
  case 2:
    return ScalarType::U32;
  case 3:
    return ScalarType::F32;
  default:
    bad("unknown scalar type");
  }
}
// A deliberately bounded public source profile: one self-contained source,
// metal_stdlib only, no macro/include/pragmas that can replace numeric options.
// This scanner only validates that boundary; the Metal compiler parses MSL.
void source_profile(const std::string &source) {
  if (source.empty() || source.size() > executable_max_source_bytes)
    bad("source is empty or exceeds byte limit");
  if (!utf8(source))
    bad("source must be valid UTF-8");
  std::string clean;
  clean.reserve(source.size());
  bool block_comment = false, line_comment = false;
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto c = static_cast<unsigned char>(source[i]);
    if (!c || (c < 32 && c != '\n' && c != '\r' && c != '\t') || c == 127)
      bad("source contains NUL or unsupported control byte");
    // Escaped newlines can hide directives from line-based validation.
    if (c == '\\')
      bad("source profile does not support backslash escapes or line splicing");
    if (line_comment) {
      if (c == '\n') {
        line_comment = false;
        clean += '\n';
      } else
        clean += ' ';
    } else if (block_comment) {
      if (c == '*' && i + 1 < source.size() && source[i + 1] == '/') {
        block_comment = false;
        clean += "  ";
        ++i;
      } else
        clean += c == '\n' ? '\n' : ' ';
    } else if (c == '/' && i + 1 < source.size() && source[i + 1] == '/') {
      line_comment = true;
      clean += "  ";
      ++i;
    } else if (c == '/' && i + 1 < source.size() && source[i + 1] == '*') {
      block_comment = true;
      clean += "  ";
      ++i;
    } else
      clean += static_cast<char>(c);
  }
  if (block_comment)
    bad("unterminated source comment");
  std::size_t begin = 0;
  while (begin < clean.size()) {
    const auto end = clean.find('\n', begin);
    auto line = clean.substr(begin, end == std::string::npos ? end : end - begin);
    line.erase(std::remove_if(line.begin(), line.end(),
                              [](char c) { return c == ' ' || c == '\t' || c == '\r'; }),
               line.end());
    if (line.find('#') != std::string::npos && line != "#include<metal_stdlib>")
      bad("source profile permits only #include <metal_stdlib> preprocessing");
    if (line.find("%:") != std::string::npos || line.find("?"
                                                          "?=") != std::string::npos)
      bad("alternate preprocessor token spellings are outside the source profile");
    if (line.find("_Pragma") != std::string::npos || line.find("__pragma") != std::string::npos)
      bad("source pragmas cannot override the numerical policy");
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
}
std::vector<std::string> tokens(const std::string &source) {
  std::vector<std::string> result;
  for (std::size_t i = 0; i < source.size();) {
    const unsigned char c = source[i];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      ++i;
      continue;
    }
    if (c == '/' && i + 1 < source.size() && source[i + 1] == '/') {
      i = source.find('\n', i + 2);
      if (i == std::string::npos)
        break;
      continue;
    }
    if (c == '/' && i + 1 < source.size() && source[i + 1] == '*') {
      const auto end = source.find("*/", i + 2);
      if (end == std::string::npos)
        bad("unterminated source comment");
      i = end + 2;
      continue;
    }
    if (result.size() == 200000)
      bad("source token count exceeds profile limit");
    if (c == '\'' || c == '"') {
      const auto end = source.find(static_cast<char>(c), i + 1);
      if (end == std::string::npos)
        bad("unterminated source literal");
      result.emplace_back("<literal>");
      i = end + 1;
      continue;
    }
    const auto start = i++;
    const auto word = [](unsigned char v) {
      return (v >= 'a' && v <= 'z') || (v >= 'A' && v <= 'Z') || (v >= '0' && v <= '9') || v == '_';
    };
    if (word(c))
      while (i < source.size() && word(source[i]))
        ++i;
    const auto token = source.substr(start, i - start);
    if (i < source.size() && source[i] == '"' &&
        (token == "R" || token == "u8R" || token == "uR" || token == "UR" || token == "LR"))
      bad("raw string literals are outside the source profile");
    result.emplace_back(token);
  }
  return result;
}
void signature_parameter(const std::vector<std::string> &p, const ExecutableEntry &entry,
                         std::set<std::uint32_t> &bindings) {
  // Public Metal reflection reports both `device const T*` and `constant T&`
  // as the same four-byte pointee. A small explicit signature grammar supplies
  // the missing ABI distinction; shader bodies remain the Metal compiler's job.
  auto annotation = std::find(p.begin(), p.end(), "[");
  const auto prefix = static_cast<std::size_t>(annotation - p.begin());
  if (annotation == p.end() || prefix < 2 || p.size() < prefix + 5 || p[prefix + 1] != "[" ||
      p[p.size() - 2] != "]" || p.back() != "]")
    bad("kernel parameters require one explicit buffer or builtin attribute");
  const auto &attribute = p[prefix + 2];
  if (attribute != "buffer") {
    static const std::set<std::string> builtins{
        "thread_position_in_grid",     "thread_position_in_threadgroup",
        "thread_index_in_threadgroup", "threadgroup_position_in_grid",
        "threads_per_threadgroup",     "threadgroups_per_grid"};
    if (p.size() != prefix + 5 || prefix != 2 || !builtins.count(attribute) ||
        (p[0] != "uint" && p[0] != "uint2" && p[0] != "uint3") || !identifier(p[1]))
      bad("unsupported builtin parameter signature");
    return;
  }
  if (p.size() != prefix + 8 || p[prefix + 3] != "(" || p[prefix + 5] != ")")
    bad("buffer binding must be one decimal literal");
  const auto &number = p[prefix + 4];
  if (number.empty() || number.size() > 2 ||
      !std::all_of(number.begin(), number.end(), [](char c) { return c >= '0' && c <= '9'; }))
    bad("buffer binding must be a decimal literal in [0, 30]");
  std::uint32_t binding = 0;
  for (char c : number)
    binding = binding * 10 + static_cast<unsigned>(c - '0');
  if (!bindings.insert(binding).second)
    bad("duplicate source buffer binding");
  const auto found = std::find_if(entry.parameters.begin(), entry.parameters.end(),
                                  [&](const auto &v) { return v.binding == binding; });
  if (found == entry.parameters.end())
    bad("manifest omits a source buffer binding");
  const auto &descriptor = *found;
  if (prefix < 4)
    bad("resource signature requires an explicit address space and pointer/reference");
  const auto &name = p[prefix - 1];
  const auto &indirection = p[prefix - 2];
  std::vector<std::string> type;
  bool is_const = false;
  for (std::size_t i = 0; i < prefix - 2; ++i) {
    if (p[i] == "const") {
      if (is_const)
        bad("duplicate const qualifier in resource signature");
      is_const = true;
    } else
      type.push_back(p[i]);
  }
  if (type.size() != 2 || (type[0] != "device" && type[0] != "constant"))
    bad("resource signatures require explicit device T* or constant T&");
  const auto scalar = type[1] == "float"  ? ScalarType::F32
                      : type[1] == "int"  ? ScalarType::I32
                      : type[1] == "uint" ? ScalarType::U32
                                          : ScalarType::Bool;
  if (scalar == ScalarType::Bool || scalar != descriptor.type || name != descriptor.name)
    bad("source resource type/name differs from manifest");
  const bool buffer = type[0] == "device";
  if ((buffer && indirection != "*") || (!buffer && indirection != "&") ||
      buffer != descriptor.buffer)
    bad("source pointer/reference kind differs from manifest or supported profile");
  const auto actual = buffer && !is_const ? ResourceAccess::ReadWrite : ResourceAccess::Read;
  if ((static_cast<unsigned>(descriptor.access) & static_cast<unsigned>(actual)) !=
      static_cast<unsigned>(actual))
    bad("manifest understates source resource access");
}
void signatures(const ExecutableModule &m) {
  const auto t = tokens(m.source);
  std::set<std::string> found;
  std::size_t scope = 0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (t[i] == "{") {
      ++scope;
      continue;
    }
    if (t[i] == "}") {
      if (scope)
        --scope;
      continue;
    }
    if (t[i] != "kernel")
      continue;
    if (scope || i + 3 >= t.size() || t[i + 1] != "void" || !identifier(t[i + 2]) ||
        t[i + 3] != "(")
      bad("entrypoints must be top-level kernel void definitions with explicit signatures");
    const auto &name = t[i + 2];
    if (!found.insert(name).second)
      bad("duplicate source entrypoint or prototype");
    auto entry = std::find_if(m.entries.begin(), m.entries.end(),
                              [&](const auto &e) { return e.name == name; });
    if (entry == m.entries.end())
      bad("source kernel missing from entrypoint manifest");
    std::set<std::uint32_t> bindings;
    std::size_t start = i + 4, depth = 0, end = start;
    for (; end < t.size(); ++end) {
      if (t[end] == "(")
        ++depth;
      else if (t[end] == ")") {
        if (depth)
          --depth;
        else
          break;
      }
      if (!depth && t[end] == ",") {
        signature_parameter({t.begin() + start, t.begin() + end}, *entry, bindings);
        start = end + 1;
      }
    }
    if (end == t.size() || end + 1 == t.size() || t[end + 1] != "{")
      bad("entrypoint prototypes or trailing signature attributes are outside the profile");
    if (start != end)
      signature_parameter({t.begin() + start, t.begin() + end}, *entry, bindings);
    else if (start != i + 4)
      bad("trailing comma in entrypoint signature");
    if (bindings.size() != entry->parameters.size())
      bad("manifest resource missing from source signature");
    i = end; // The next iteration enters the ordinary compiled function body.
  }
  if (found.size() != m.entries.size())
    bad("manifest entrypoint missing from source definitions");
}
struct Writer {
  std::vector<unsigned char> bytes;
  void u32(std::uint32_t v) {
    if (bytes.size() > executable_max_bytes - 4)
      bad("artifact exceeds byte limit");
    for (unsigned i = 0; i < 32; i += 8)
      bytes.push_back(static_cast<unsigned char>(v >> i));
  }
  void u64(std::uint64_t v) {
    u32(static_cast<std::uint32_t>(v));
    u32(v >> 32);
  }
  void string(const std::string &s) {
    u32(static_cast<std::uint32_t>(s.size()));
    if (s.size() > executable_max_bytes - bytes.size())
      bad("artifact exceeds byte limit");
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};
struct Reader {
  const unsigned char *data;
  std::size_t size, offset = 0;
  Reader(const void *d, std::size_t n) : data(static_cast<const unsigned char *>(d)), size(n) {
    if (!d || n > executable_max_bytes)
      bad("null input or excessive byte count");
  }
  void need(std::size_t n) const {
    if (n > size - offset)
      bad("truncated input");
  }
  std::uint32_t u32() {
    need(4);
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 32; i += 8)
      v |= std::uint32_t(data[offset++]) << i;
    return v;
  }
  std::uint64_t u64() {
    const auto low = u32();
    return std::uint64_t(low) | (std::uint64_t(u32()) << 32);
  }
  std::uint32_t count(std::uint32_t limit, std::size_t minimum) {
    const auto n = u32();
    if (n > limit || n > (size - offset) / minimum)
      bad("excessive or truncated collection");
    return n;
  }
  std::string string(std::uint32_t limit = max_text) {
    auto n = count(limit, 1);
    std::string s(reinterpret_cast<const char *>(data + offset), n);
    offset += n;
    return s;
  }
};
std::uint32_t rotate(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
} // namespace

std::string source_sha256(const std::string &source) {
  // SHA-256, FIPS 180-4. Used for reproducible provenance, not authentication.
  constexpr std::uint32_t k[] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
      0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
      0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
      0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
      0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  std::array<std::uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::vector<unsigned char> bytes(source.begin(), source.end());
  bytes.push_back(0x80);
  while (bytes.size() % 64 != 56)
    bytes.push_back(0);
  const auto bit_count = std::uint64_t(source.size()) * 8;
  for (int i = 56; i >= 0; i -= 8)
    bytes.push_back(static_cast<unsigned char>(bit_count >> i));
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
    std::uint32_t w[64]{};
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned j = 0; j < 4; ++j)
        w[i] = (w[i] << 8) | bytes[offset + i * 4 + j];
    for (unsigned i = 16; i < 64; ++i) {
      const auto x = w[i - 15], y = w[i - 2];
      w[i] = w[i - 16] + (rotate(x, 7) ^ rotate(x, 18) ^ (x >> 3)) + w[i - 7] +
             (rotate(y, 17) ^ rotate(y, 19) ^ (y >> 10));
    }
    auto a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], z = h[7];
    for (unsigned i = 0; i < 64; ++i) {
      const auto t1 =
          z + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const auto t2 =
          (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      z = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += z;
  }
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto value : h)
    for (int shift = 28; shift >= 0; shift -= 4)
      result += hex[(value >> shift) & 15];
  return result;
}

void verify_executable(const ExecutableModule &m) {
  if (m.format != ExecutableFormat::MslSource || m.target != "metal-msl3.1")
    bad("unsupported executable format or target");
  if (m.numerical_policy != 1)
    bad("unsupported numerical policy");
  text(m.producer);
  text(m.producer_version);
  text(m.source_name);
  text(m.source_sha256);
  source_profile(m.source);
  if (m.source_sha256 != source_sha256(m.source))
    bad("source SHA-256 mismatch");
  if (m.entries.empty() || m.entries.size() > max_entries)
    bad("entrypoint count outside limits");
  std::set<std::string> entries;
  for (const auto &e : m.entries) {
    if (!identifier(e.name) || !entries.insert(e.name).second)
      bad("invalid or duplicate entrypoint name");
    const bool any = std::any_of(e.required_block.begin(), e.required_block.end(),
                                 [](auto n) { return n != 0; });
    std::uint64_t product = 1;
    for (auto n : e.required_block) {
      if (any && (!n || n > 1024))
        bad("invalid required block dimensions");
      product *= n;
    }
    if (any && product > 1024)
      bad("required block exceeds profile limit");
    if (e.parameters.size() > max_parameters)
      bad("parameter count exceeds Metal binding profile");
    std::set<std::string> names;
    std::set<std::uint32_t> slots;
    for (const auto &p : e.parameters) {
      if (!identifier(p.name) || !names.insert(p.name).second)
        bad("invalid or duplicate parameter name");
      (void)scalar_code(p.type);
      if (p.binding >= max_parameters || !slots.insert(p.binding).second)
        bad("invalid or duplicate buffer binding");
      if (p.access != ResourceAccess::Read && p.access != ResourceAccess::Write &&
          p.access != ResourceAccess::ReadWrite)
        bad("unknown resource access");
      if (p.alignment < 4 || p.alignment > 4096 || (p.alignment & (p.alignment - 1)))
        bad("invalid resource alignment");
      if (!p.minimum_bytes || p.minimum_bytes % 4)
        bad("invalid resource minimum size");
      if (!p.buffer &&
          (p.access != ResourceAccess::Read || p.minimum_bytes != 4 || p.alignment != 4))
        bad("scalar parameters require read-only, four-byte values");
    }
  }
  signatures(m);
}

std::vector<unsigned char> serialize_executable(const ExecutableModule &m) {
  verify_executable(m);
  Writer w;
  w.bytes.insert(w.bytes.end(), std::begin(magic), std::end(magic));
  w.u32(version);
  w.u32(static_cast<std::uint32_t>(m.format));
  w.string(m.target);
  w.u32(m.numerical_policy);
  w.string(m.producer);
  w.string(m.producer_version);
  w.string(m.source_name);
  w.string(m.source_sha256);
  w.string(m.source);
  w.u32(static_cast<std::uint32_t>(m.entries.size()));
  for (const auto &e : m.entries) {
    w.string(e.name);
    for (auto n : e.required_block)
      w.u32(n);
    w.u32(static_cast<std::uint32_t>(e.parameters.size()));
    for (const auto &p : e.parameters) {
      w.string(p.name);
      w.u32(scalar_code(p.type));
      w.u32(p.buffer ? 1 : 0);
      w.u32(static_cast<std::uint32_t>(p.access));
      w.u32(p.binding);
      w.u32(p.alignment);
      w.u64(p.minimum_bytes);
    }
  }
  return std::move(w.bytes);
}

bool is_executable_artifact(const void *data, std::size_t size) noexcept {
  return data && size >= sizeof(magic) && std::memcmp(data, magic, sizeof(magic)) == 0;
}

ExecutableModule deserialize_executable(const void *data, std::size_t size) {
  Reader r(data, size);
  if (!is_executable_artifact(data, size))
    bad("incorrect magic");
  r.offset = sizeof(magic);
  if (r.u32() != version)
    bad("unsupported container version");
  ExecutableModule m;
  m.format = static_cast<ExecutableFormat>(r.u32());
  m.target = r.string();
  m.numerical_policy = r.u32();
  m.producer = r.string();
  m.producer_version = r.string();
  m.source_name = r.string();
  m.source_sha256 = r.string();
  m.source = r.string(executable_max_source_bytes);
  auto count = r.count(max_entries, 20);
  for (std::uint32_t i = 0; i < count; ++i) {
    ExecutableEntry e;
    e.name = r.string();
    for (auto &n : e.required_block)
      n = r.u32();
    auto parameters = r.count(max_parameters, 32);
    for (std::uint32_t j = 0; j < parameters; ++j) {
      ExecutableParameter p;
      p.name = r.string();
      p.type = scalar_type(r.u32());
      const auto buffer = r.u32();
      if (buffer > 1)
        bad("unknown parameter kind");
      p.buffer = buffer;
      p.access = static_cast<ResourceAccess>(r.u32());
      p.binding = r.u32();
      p.alignment = r.u32();
      p.minimum_bytes = r.u64();
      e.parameters.push_back(std::move(p));
    }
    m.entries.push_back(std::move(e));
  }
  if (r.offset != r.size)
    bad("trailing data");
  verify_executable(m);
  return m;
}
} // namespace paralyn
