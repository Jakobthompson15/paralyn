#include "kernel_case.hpp"
#include "paralyn/executable.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#define TOML_ENABLE_FORMATTERS 0
#include <tomlplusplus/toml.hpp>

namespace paralyn::cli {
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;
namespace {
constexpr std::size_t toml_limit = 16 * 1024 * 1024;
constexpr std::size_t data_limit = 256 * 1024 * 1024;
constexpr std::uint64_t max_elements = data_limit / 4;

// Errors carry a schema-specific prefix (P-CASE-* or P-PROJECT-*).
struct Schema {
  std::string prefix, file;
  [[noreturn]] void fail(const std::string &kind, const std::string &message,
                         const toml::node *node = nullptr) const {
    std::string where = file;
    if (node && node->source().begin.line)
      where += ":" + std::to_string(node->source().begin.line) + ":" +
               std::to_string(node->source().begin.column);
    throw Diagnostic(prefix + "-" + kind, "input", where + ": " + message);
  }
  void only(const toml::table &table, std::initializer_list<const char *> allowed,
            const std::string &where) const {
    for (auto &&[key, value] : table) {
      bool known = false;
      for (auto name : allowed)
        known |= key.str() == name;
      if (!known)
        fail("UNKNOWN-FIELD",
             "unknown field " + (where.empty() ? "" : where + ".") + std::string(key.str()) +
                 "; this schema version accepts only its documented fields",
             &value);
    }
  }
  const toml::node &need(const toml::table &table, const char *key,
                         const std::string &where) const {
    auto node = table.get(key);
    if (!node)
      fail("MISSING-FIELD", "missing required field " + where + "." + key, &table);
    return *node;
  }
  const toml::table &table(const toml::node &node, const std::string &where) const {
    if (!node.is_table())
      fail("TYPE", where + " must be a table", &node);
    return *node.as_table();
  }
  std::string text(const toml::node &node, const std::string &where) const {
    if (!node.is_string())
      fail("TYPE", where + " must be a string", &node);
    auto value = node.as_string()->get();
    if (value.empty())
      fail("VALUE", where + " must not be empty", &node);
    return value;
  }
  bool boolean(const toml::node &node, const std::string &where) const {
    if (!node.is_boolean())
      fail("TYPE", where + " must be true or false", &node);
    return node.as_boolean()->get();
  }
  std::int64_t integer(const toml::node &node, const std::string &where, std::int64_t low,
                       std::int64_t high) const {
    if (!node.is_integer())
      fail("TYPE", where + " must be an integer", &node);
    auto value = node.as_integer()->get();
    if (value < low || value > high)
      fail("VALUE",
           where + " is out of range [" + std::to_string(low) + ", " + std::to_string(high) + "]",
           &node);
    return value;
  }
  double number(const toml::node &node, const std::string &where) const {
    if (node.is_integer())
      return static_cast<double>(node.as_integer()->get());
    if (!node.is_floating_point())
      fail("TYPE", where + " must be a number", &node);
    return node.as_floating_point()->get();
  }
  // Declared data is converted without silent rounding or wrapping.
  std::uint32_t word(const toml::node &node, DType type, const std::string &where) const {
    std::uint32_t bits = 0;
    if (type == DType::F32) {
      if (!node.is_integer() && !node.is_floating_point())
        fail("TYPE", where + " must be an f32 number", &node);
      double value = number(node, where);
      float narrowed = static_cast<float>(value);
      if (node.is_integer() && (node.as_integer()->get() > (std::int64_t(1) << 53) ||
                                node.as_integer()->get() < -(std::int64_t(1) << 53)))
        fail("VALUE", where + " integer is not exactly representable as f32", &node);
      if (!std::isnan(value) && static_cast<double>(narrowed) != value)
        fail("VALUE",
             where + " is not exactly representable as f32; use a binary data file for "
                     "values that require rounding",
             &node);
      std::memcpy(&bits, &narrowed, 4);
    } else if (type == DType::I32) {
      auto value = static_cast<std::int32_t>(integer(node, where, INT32_MIN, INT32_MAX));
      std::memcpy(&bits, &value, 4);
    } else
      bits = static_cast<std::uint32_t>(integer(node, where, 0, UINT32_MAX));
    return bits;
  }
  DType dtype(const toml::node &node, const std::string &where) const {
    auto name = text(node, where);
    if (name == "f32")
      return DType::F32;
    if (name == "i32")
      return DType::I32;
    if (name == "u32")
      return DType::U32;
    fail("VALUE", where + " must be one of f32, i32, u32 (got " + name + ")", &node);
  }
  std::array<std::uint32_t, 3> dim3(const toml::node &node, const std::string &where) const {
    if (!node.is_array() || node.as_array()->size() != 3)
      fail("TYPE", where + " must be an array of exactly three positive integers", &node);
    std::array<std::uint32_t, 3> out{};
    for (std::size_t i = 0; i < 3; ++i)
      out[i] = static_cast<std::uint32_t>(
          integer(*node.as_array()->get(i), where + "[" + std::to_string(i) + "]", 1, UINT32_MAX));
    return out;
  }
  void header(const toml::table &root, const char *schema_name) const {
    auto schema = text(need(root, "schema", ""), "schema");
    if (schema != schema_name)
      fail("SCHEMA", std::string("schema must be \"") + schema_name + "\"",
           root.get("schema"));
    if (integer(need(root, "schema_version", ""), "schema_version", 0, INT32_MAX) != 1)
      fail("VERSION", "unsupported schema_version; this Paralyn reads version 1",
           root.get("schema_version"));
  }
  toml::table parse(const fs::path &path, std::string &sha, std::string *kept = nullptr) const {
    auto bytes = read_bounded(path, toml_limit, prefix + "-FILE");
    sha = sha256_bytes(bytes.data(), bytes.size());
    if (kept)
      *kept = bytes;
    try {
      return toml::parse(bytes, path.string());
    } catch (const toml::parse_error &error) {
      throw Diagnostic(prefix + "-SYNTAX", "input",
                       path.string() + ":" + std::to_string(error.source().begin.line) + ":" +
                           std::to_string(error.source().begin.column) +
                           ": invalid TOML: " + std::string(error.description()));
    }
  }
};

fs::path resolve(const fs::path &base, const std::string &value) {
  fs::path p(value);
  return fs::absolute(p.is_absolute() ? p : base / p).lexically_normal();
}
bool is_sha256(const std::string &text) {
  return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
std::vector<std::uint32_t> data_file(const Schema &schema, const fs::path &path,
                                     const std::string &declared, std::uint64_t length,
                                     const toml::node *node, const std::string &where) {
  auto bytes = read_bounded(path, data_limit, schema.prefix + "-DATA-FILE");
  auto actual = sha256_bytes(bytes.data(), bytes.size());
  if (actual != declared)
    throw Diagnostic(schema.prefix + "-DATA-HASH", "input",
                     where + ": " + path.string() + " has SHA256 " + actual +
                         " but the case declares " + declared);
  if (bytes.size() != length * 4)
    schema.fail("DATA-SIZE",
                where + ": " + path.string() + " holds " + std::to_string(bytes.size()) +
                    " bytes; " + std::to_string(length) + " little-endian 32-bit elements need " +
                    std::to_string(length * 4),
                node);
  std::vector<std::uint32_t> words(length);
  for (std::uint64_t i = 0; i < length; ++i) {
    std::uint32_t w = 0;
    for (unsigned b = 0; b < 4; ++b)
      w |= std::uint32_t(static_cast<unsigned char>(bytes[i * 4 + b])) << (8 * b);
    words[i] = w;
  }
  return words;
}
// Exactly one data origin: inline values, an explicit fill, or a hashed file.
std::vector<std::uint32_t> declared_data(const Schema &schema, const toml::table &t,
                                         const fs::path &base, DType type, std::uint64_t length,
                                         const std::string &where, std::string &origin,
                                         fs::path &file, std::string &sha, bool allow_fill) {
  int sources = !!t.get("values") + !!t.get("file") + (allow_fill && t.get("fill"));
  if (sources != 1)
    schema.fail("DATA-SOURCE",
                where + " must declare exactly one of values, file" +
                    (allow_fill ? ", fill" : "") + "; Paralyn never invents data",
                &t);
  if (!t.get("file") && t.get("sha256"))
    schema.fail("UNKNOWN-FIELD", where + ".sha256 is only valid with file", t.get("sha256"));
  std::vector<std::uint32_t> words;
  if (auto values = t.get("values")) {
    origin = "values";
    if (!values->is_array())
      schema.fail("TYPE", where + ".values must be an array", values);
    auto &array = *values->as_array();
    if (array.size() != length)
      schema.fail("DATA-SIZE",
                  where + ".values has " + std::to_string(array.size()) +
                      " elements; declared length is " + std::to_string(length),
                  values);
    for (std::size_t i = 0; i < array.size(); ++i)
      words.push_back(
          schema.word(*array.get(i), type, where + ".values[" + std::to_string(i) + "]"));
  } else if (auto fill = t.get("fill")) {
    origin = "fill";
    words.assign(length, schema.word(*fill, type, where + ".fill"));
  } else {
    origin = "file";
    file = resolve(base, schema.text(*t.get("file"), where + ".file"));
    sha = schema.text(schema.need(t, "sha256", where), where + ".sha256");
    if (!is_sha256(sha))
      schema.fail("VALUE", where + ".sha256 must be 64 lowercase hexadecimal digits",
                  t.get("sha256"));
    words = data_file(schema, file, sha, length, t.get("file"), where);
  }
  return words;
}
Tolerance tolerance(const Schema &schema, const toml::node &node, DType type,
                    const std::string &where) {
  auto &t = schema.table(node, where);
  auto kind = schema.text(schema.need(t, "kind", where), where + ".kind");
  Tolerance out;
  if (kind == "exact") {
    schema.only(t, {"kind"}, where);
    out.kind = Tolerance::Kind::Exact;
  } else if (kind == "absolute_relative") {
    schema.only(t, {"kind", "absolute", "relative"}, where);
    out.kind = Tolerance::Kind::AbsoluteRelative;
    out.absolute = schema.number(schema.need(t, "absolute", where), where + ".absolute");
    out.relative = schema.number(schema.need(t, "relative", where), where + ".relative");
    if (!(out.absolute >= 0) || !(out.relative >= 0) || std::isinf(out.absolute) ||
        std::isinf(out.relative))
      schema.fail("VALUE", where + " bounds must be finite and nonnegative", &node);
  } else if (kind == "ulp") {
    schema.only(t, {"kind", "ulp"}, where);
    out.kind = Tolerance::Kind::Ulp;
    out.ulp = static_cast<std::uint64_t>(
        schema.integer(schema.need(t, "ulp", where), where + ".ulp", 0, INT32_MAX));
  } else
    schema.fail("VALUE", where + ".kind must be exact, absolute_relative or ulp", &node);
  if (type != DType::F32 && out.kind != Tolerance::Kind::Exact)
    schema.fail("VALUE", where + ": integer outputs only support kind = \"exact\"", &node);
  return out;
}
struct Builtin {
  const char *name;
  std::vector<std::pair<const char *, bool>> roles; // role, is_buffer
  std::vector<const char *> constants;
};
const std::vector<Builtin> &builtins() {
  static const std::vector<Builtin> table{
      {"vector-add-f32", {{"lhs", true}, {"rhs", true}, {"count", false}}, {}},
      {"block-sum-f32", {{"input", true}, {"count", false}}, {"group_size"}},
      {"transpose-f32", {{"input", true}, {"width", false}, {"height", false}}, {}},
  };
  return table;
}
float as_float(std::uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}
std::uint32_t as_word(float f) {
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  return w;
}
std::int64_t ordered(std::uint32_t w) {
  return (w & 0x80000000u) ? -static_cast<std::int64_t>(w & 0x7fffffffu)
                           : static_cast<std::int64_t>(w);
}
Json display(DType type, std::uint32_t w) {
  if (type == DType::F32) {
    float f = as_float(w);
    if (std::isfinite(f))
      return f;
    return std::isnan(f) ? "nan" : (f > 0 ? "inf" : "-inf");
  }
  if (type == DType::I32) {
    std::int32_t i;
    std::memcpy(&i, &w, 4);
    return i;
  }
  return w;
}
} // namespace

const char *dtype_name(DType type) {
  return type == DType::F32 ? "f32" : type == DType::I32 ? "i32" : "u32";
}
std::string sha256_bytes(const void *data, std::size_t size) {
  return paralyn::source_sha256(std::string(static_cast<const char *>(data), size));
}
std::string read_bounded(const fs::path &path, std::size_t limit, const std::string &error_id) {
  std::error_code error;
  if (!fs::is_regular_file(path, error))
    throw Diagnostic(error_id, "input", "Cannot read " + path.string() + ": not a regular file");
  auto size = fs::file_size(path, error);
  if (error || size > limit)
    throw Diagnostic(error_id, "input",
                     "Cannot read " + path.string() + ": size exceeds " + std::to_string(limit) +
                         " bytes or is unavailable");
  std::ifstream file(path, std::ios::binary);
  std::string text(static_cast<std::size_t>(size), '\0');
  if (!file || !file.read(text.data(), static_cast<std::streamsize>(size)))
    throw Diagnostic(error_id, "input", "Cannot read complete file " + path.string());
  return text;
}

// Static shape contract of a builtin reference: every fact it needs (buffer
// lengths, scalar values, constants) is declared in the case, so violations are
// rejected at load time, before any module load or GPU submission. Returns an
// empty string when the builtin can be evaluated for the declared case.
static std::string builtin_shape_problem(const KernelCase &kc, const CaseCheck &check) {
  if (check.reference_kind != "builtin")
    return "";
  auto buffer = [&](const std::string &role) -> const CaseBuffer * {
    for (const auto &b : kc.buffers)
      if (b.name == check.roles.at(role))
        return &b;
    return nullptr;
  };
  const CaseBuffer *out = nullptr;
  for (const auto &b : kc.buffers)
    if (b.name == check.buffer)
      out = &b;
  std::string problem;
  auto scalar = [&](const std::string &role) -> std::uint64_t {
    for (const auto &s : kc.scalars)
      if (s.name == check.roles.at(role)) {
        if (s.type == DType::I32 && (s.bits & 0x80000000u) && problem.empty())
          problem = "role " + role + " (" + s.name + ") is a negative i32";
        return s.bits;
      }
    if (problem.empty())
      problem = "role " + role + " is unbound";
    return 0;
  };
  if (!out)
    return "output buffer " + check.buffer + " is not declared";
  if (check.builtin == "vector-add-f32") {
    auto lhs = buffer("lhs"), rhs = buffer("rhs");
    auto n = scalar("count");
    if (!problem.empty())
      return problem;
    if (!lhs || !rhs || n > lhs->length || n > rhs->length || n > out->length)
      return "count (" + std::to_string(n) + ") exceeds the length of lhs, rhs or " + check.buffer;
  } else if (check.builtin == "block-sum-f32") {
    auto input = buffer("input");
    auto n = scalar("count");
    if (!problem.empty())
      return problem;
    auto group = check.constants.at("group_size");
    auto groups = (n + group - 1) / group;
    if (!input || n > input->length || groups > out->length)
      return "count (" + std::to_string(n) + ") exceeds the input length or " + check.buffer +
             " has fewer than " + std::to_string(groups) + " group elements";
  } else if (check.builtin == "transpose-f32") {
    auto input = buffer("input");
    std::uint64_t width = scalar("width"), height = scalar("height");
    if (!problem.empty())
      return problem;
    if (!input || width * height > input->length || width * height > out->length)
      return "width*height (" + std::to_string(width * height) +
             ") exceeds the length of input or " + check.buffer;
  }
  return "";
}

KernelCase load_case(const fs::path &input) {
  KernelCase out;
  out.path = fs::absolute(input).lexically_normal();
  Schema schema{"P-CASE", out.path.string()};
  auto root = schema.parse(out.path, out.sha256, &out.bytes);
  const auto base = out.path.parent_path();
  schema.only(root, {"schema", "schema_version", "case", "launch", "scalars", "buffers", "verify"},
              "");
  schema.header(root, "paralyn.kernel-case");
  auto &info = schema.table(schema.need(root, "case", ""), "case");
  schema.only(info, {"name", "entry", "description"}, "case");
  out.name = schema.text(schema.need(info, "name", "case"), "case.name");
  if (auto entry = info.get("entry"))
    out.entry = schema.text(*entry, "case.entry");
  if (auto description = info.get("description"))
    out.description = schema.text(*description, "case.description");

  auto &launch = schema.table(schema.need(root, "launch", ""), "launch");
  schema.only(launch, {"grid", "block"}, "launch");
  out.grid = schema.dim3(schema.need(launch, "grid", "launch"), "launch.grid");
  out.block = schema.dim3(schema.need(launch, "block", "launch"), "launch.block");

  std::set<std::string> names;
  if (auto scalars = root.get("scalars"))
    for (auto &&[key, node] : schema.table(*scalars, "scalars")) {
      const std::string name(key.str()), where = "scalars." + name;
      auto &t = schema.table(node, where);
      schema.only(t, {"type", "value"}, where);
      CaseScalar s;
      s.name = name;
      s.type = schema.dtype(schema.need(t, "type", where), where + ".type");
      s.bits = schema.word(schema.need(t, "value", where), s.type, where + ".value");
      names.insert(name);
      out.scalars.push_back(s);
    }
  auto &buffers = schema.table(schema.need(root, "buffers", ""), "buffers");
  for (auto &&[key, node] : buffers) {
    const std::string name(key.str()), where = "buffers." + name;
    auto &t = schema.table(node, where);
    schema.only(t, {"dtype", "length", "values", "fill", "file", "sha256", "output"}, where);
    if (!names.insert(name).second)
      schema.fail("DUPLICATE-ARGUMENT", name + " is declared as both a scalar and a buffer",
                  &node);
    CaseBuffer b;
    b.name = name;
    b.type = schema.dtype(schema.need(t, "dtype", where), where + ".dtype");
    b.length = static_cast<std::uint64_t>(schema.integer(
        schema.need(t, "length", where), where + ".length", 1, std::int64_t(max_elements)));
    if (auto output = t.get("output"))
      b.output = schema.boolean(*output, where + ".output");
    b.initial = declared_data(schema, t, base, b.type, b.length, where, b.origin, b.file,
                              b.sha256, true);
    out.buffers.push_back(std::move(b));
  }
  if (std::none_of(out.buffers.begin(), out.buffers.end(),
                   [](const CaseBuffer &b) { return b.output; }))
    schema.fail("OUTPUT-REQUIRED",
                "declare at least one buffer with output = true; a case without an observable "
                "output cannot be run or verified",
                &buffers);
  auto find_buffer = [&](const std::string &name) -> const CaseBuffer * {
    for (const auto &b : out.buffers)
      if (b.name == name)
        return &b;
    return nullptr;
  };
  auto find_scalar = [&](const std::string &name) -> const CaseScalar * {
    for (const auto &s : out.scalars)
      if (s.name == name)
        return &s;
    return nullptr;
  };
  if (auto verify = root.get("verify")) {
    if (!verify->is_array_of_tables())
      schema.fail("TYPE", "verify must be an array of tables ([[verify]])", verify);
    std::set<std::string> checked;
    std::size_t index = 0;
    for (auto &node : *verify->as_array()) {
      const std::string where = "verify[" + std::to_string(index++) + "]";
      auto &t = *node.as_table();
      schema.only(t, {"buffer", "reference", "tolerance"}, where);
      CaseCheck c;
      c.buffer = schema.text(schema.need(t, "buffer", where), where + ".buffer");
      auto buffer = find_buffer(c.buffer);
      if (!buffer || !buffer->output)
        schema.fail("VERIFY-BUFFER", where + ".buffer must name a declared output buffer",
                    t.get("buffer"));
      if (!checked.insert(c.buffer).second)
        schema.fail("VERIFY-BUFFER", where + ": " + c.buffer + " already has a verification",
                    t.get("buffer"));
      auto &reference = schema.table(schema.need(t, "reference", where), where + ".reference");
      const std::string rwhere = where + ".reference";
      if (auto builtin = reference.get("builtin")) {
        c.reference_kind = "builtin";
        c.builtin = schema.text(*builtin, rwhere + ".builtin");
        const Builtin *spec = nullptr;
        for (const auto &b : builtins())
          if (c.builtin == b.name)
            spec = &b;
        if (!spec) {
          std::string known;
          for (const auto &b : builtins())
            known += std::string(known.empty() ? "" : ", ") + b.name;
          throw Diagnostic("P-REFERENCE-BUILTIN-UNKNOWN", "input",
                           out.path.string() + ": unknown builtin reference " + c.builtin +
                               "; registered builtins: " + known);
        }
        for (auto &&[key, value] : reference) {
          std::string k(key.str());
          bool known = k == "builtin";
          for (auto &role : spec->roles)
            known |= k == role.first;
          for (auto constant : spec->constants)
            known |= k == constant;
          if (!known)
            schema.fail("UNKNOWN-FIELD", rwhere + "." + k + " is not a parameter of builtin " +
                                             c.builtin,
                        &value);
        }
        for (auto &[role, is_buffer] : spec->roles) {
          auto name = schema.text(schema.need(reference, role, rwhere), rwhere + "." + role);
          if (is_buffer ? (!find_buffer(name) || find_buffer(name)->type != DType::F32)
                        : (!find_scalar(name) || find_scalar(name)->type == DType::F32))
            schema.fail("VERIFY-ROLE",
                        rwhere + "." + role + " must name a declared " +
                            (is_buffer ? "f32 buffer" : "u32 or i32 scalar"),
                        reference.get(role));
          c.roles[role] = name;
        }
        for (auto constant : spec->constants)
          c.constants[constant] = static_cast<std::uint64_t>(schema.integer(
              schema.need(reference, constant, rwhere), rwhere + "." + constant, 1, 1 << 20));
        if (buffer->type != DType::F32)
          schema.fail("VERIFY-ROLE", where + ": builtin " + c.builtin + " produces f32 output",
                      t.get("buffer"));
      } else {
        schema.only(reference, {"values", "file", "sha256"}, rwhere);
        std::string origin;
        c.expected = declared_data(schema, reference, base, buffer->type, buffer->length, rwhere,
                                   origin, c.file, c.sha256, false);
        c.reference_kind = origin;
      }
      c.tolerance =
          tolerance(schema, schema.need(t, "tolerance", where), buffer->type, where + ".tolerance");
      out.checks.push_back(std::move(c));
    }
  }
  for (const auto &check : out.checks) {
    auto problem = builtin_shape_problem(out, check);
    if (!problem.empty())
      throw Diagnostic("P-REFERENCE-SHAPE", "input",
                       out.path.string() + ": builtin " + check.builtin +
                           " cannot be evaluated for " + check.buffer + ": " + problem +
                           ". The declared case would make the kernel or the reference access "
                           "elements outside the declared buffers; fix lengths or scalars");
  }
  return out;
}

CheckResult evaluate_check(const KernelCase &kc, const CaseCheck &check,
                           const std::vector<std::uint32_t> &actual) {
  const CaseBuffer *out = nullptr;
  for (const auto &b : kc.buffers)
    if (b.name == check.buffer)
      out = &b;
  auto buffer = [&](const std::string &role) -> const CaseBuffer & {
    for (const auto &b : kc.buffers)
      if (b.name == check.roles.at(role))
        return b;
    throw Diagnostic("P-CASE-VERIFY-ROLE", "verification", "unbound role " + role);
  };
  auto shape_error = [&](const std::string &message) {
    return Diagnostic("P-REFERENCE-SHAPE", "verification",
                      "builtin " + check.builtin + " cannot be evaluated for " + check.buffer +
                          ": " + message);
  };
  auto scalar = [&](const std::string &role) -> std::uint32_t {
    for (const auto &s : kc.scalars)
      if (s.name == check.roles.at(role)) {
        if (s.type == DType::I32 && (s.bits & 0x80000000u))
          throw shape_error("role " + role + " is a negative i32");
        return s.bits;
      }
    throw Diagnostic("P-CASE-VERIFY-ROLE", "verification", "unbound role " + role);
  };
  // Elements a builtin does not define keep the declared initial contents: the
  // kernel must not write them, so canaries/sentinels are checked too.
  std::vector<std::uint32_t> expected = check.expected;
  if (auto problem = builtin_shape_problem(kc, check); !problem.empty())
    throw shape_error(problem); // Unreachable for cases from load_case.
  if (check.reference_kind == "builtin") {
    expected = out->initial;
    if (check.builtin == "vector-add-f32") {
      auto &lhs = buffer("lhs");
      auto &rhs = buffer("rhs");
      auto n = scalar("count");
      if (n > lhs.length || n > rhs.length || n > out->length)
        throw shape_error("count exceeds a buffer length");
      for (std::uint32_t i = 0; i < n; ++i)
        expected[i] = as_word(as_float(lhs.initial[i]) + as_float(rhs.initial[i]));
    } else if (check.builtin == "block-sum-f32") {
      auto &input = buffer("input");
      auto n = scalar("count");
      auto group = check.constants.at("group_size");
      auto groups = (std::uint64_t(n) + group - 1) / group;
      if (n > input.length || groups > out->length)
        throw shape_error("count exceeds the input length or the output has too few groups");
      for (std::uint64_t g = 0; g < groups; ++g) {
        double sum = 0; // Sequential double accumulation; order differs from the GPU tree.
        for (std::uint64_t i = g * group; i < std::min<std::uint64_t>(n, (g + 1) * group); ++i)
          sum += as_float(input.initial[i]);
        expected[g] = as_word(static_cast<float>(sum));
      }
    } else if (check.builtin == "transpose-f32") {
      auto &input = buffer("input");
      std::uint64_t width = scalar("width"), height = scalar("height");
      if (width * height > input.length || width * height > out->length)
        throw shape_error("width*height exceeds a buffer length");
      for (std::uint64_t y = 0; y < height; ++y)
        for (std::uint64_t x = 0; x < width; ++x)
          expected[x * height + y] = input.initial[y * width + x];
    }
  }
  CheckResult result;
  Json mismatches = Json::array();
  std::uint64_t failures = 0, max_ulp = 0;
  double max_abs = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto a = actual[i], e = expected[i];
    bool ok = a == e;
    if (out->type == DType::F32) {
      float fa = as_float(a), fe = as_float(e);
      if (std::isnan(fa) || std::isnan(fe))
        // exact: identical bits; toleranced kinds: any NaN matches only a NaN.
        ok = check.tolerance.kind == Tolerance::Kind::Exact ? a == e
                                                            : std::isnan(fa) && std::isnan(fe);
      else if (check.tolerance.kind != Tolerance::Kind::Exact || a != e) {
        auto distance = static_cast<std::uint64_t>(std::llabs(ordered(a) - ordered(e)));
        double diff = std::fabs(double(fa) - double(fe));
        max_ulp = std::max(max_ulp, distance);
        if (std::isfinite(diff))
          max_abs = std::max(max_abs, diff);
        if (check.tolerance.kind == Tolerance::Kind::Ulp)
          ok = distance <= check.tolerance.ulp;
        else if (check.tolerance.kind == Tolerance::Kind::AbsoluteRelative)
          ok = fa == fe || diff <= check.tolerance.absolute +
                                       check.tolerance.relative * std::fabs(double(fe));
      }
    }
    if (!ok) {
      if (mismatches.size() < 16)
        mismatches.push_back(
            {{"index", i}, {"expected", display(out->type, e)}, {"actual", display(out->type, a)}});
      ++failures;
    }
  }
  result.compared = expected.size();
  result.passed = failures == 0;
  Json tol;
  switch (check.tolerance.kind) {
  case Tolerance::Kind::Exact:
    tol = {{"kind", "exact"}};
    break;
  case Tolerance::Kind::AbsoluteRelative:
    tol = {{"kind", "absolute_relative"},
           {"absolute", check.tolerance.absolute},
           {"relative", check.tolerance.relative}};
    break;
  case Tolerance::Kind::Ulp:
    tol = {{"kind", "ulp"}, {"ulp", check.tolerance.ulp}};
    break;
  }
  Json reference = {{"kind", check.reference_kind}};
  if (check.reference_kind == "builtin") {
    reference["builtin"] = check.builtin;
    reference["roles"] = check.roles;
    if (!check.constants.empty())
      reference["constants"] = check.constants;
    reference["computed_by"] = "Paralyn host CPU reference from declared case inputs";
  } else if (check.reference_kind == "file") {
    reference["file"] = check.file.string();
    reference["sha256"] = check.sha256;
  }
  result.record = {{"buffer", check.buffer},
                   {"dtype", dtype_name(out->type)},
                   {"reference", reference},
                   {"tolerance", tol},
                   {"compared", result.compared},
                   {"mismatches", failures},
                   {"status", result.passed ? "passed" : "failed"},
                   {"first_mismatches", mismatches}};
  if (out->type == DType::F32)
    result.record["max_abs_error"] = max_abs, result.record["max_ulp_distance"] = max_ulp;
  return result;
}

Project load_project(const fs::path &input) {
  Project out;
  out.path = fs::absolute(input).lexically_normal();
  Schema schema{"P-PROJECT", out.path.string()};
  auto root = schema.parse(out.path, out.sha256);
  const auto base = out.path.parent_path();
  schema.only(root, {"schema", "schema_version", "project", "modules", "cases", "programs"}, "");
  schema.header(root, "paralyn.project");
  auto &info = schema.table(schema.need(root, "project", ""), "project");
  schema.only(info, {"name", "default"}, "project");
  out.name = schema.text(schema.need(info, "name", "project"), "project.name");
  if (auto d = info.get("default"))
    out.default_target = schema.text(*d, "project.default");
  if (auto modules = root.get("modules"))
    for (auto &&[key, node] : schema.table(*modules, "modules")) {
      const std::string name(key.str()), where = "modules." + name;
      auto &t = schema.table(node, where);
      schema.only(t, {"source", "manifest"}, where);
      ProjectModule m;
      m.name = name;
      m.source = resolve(base, schema.text(schema.need(t, "source", where), where + ".source"));
      auto ext = m.source.extension().string();
      if (ext == ".metal") {
        m.manifest =
            resolve(base, schema.text(schema.need(t, "manifest", where), where + ".manifest"));
      } else if (ext == ".prx" || ext == ".prk") {
        if (t.get("manifest"))
          schema.fail("UNKNOWN-FIELD", where + ".manifest is only valid for .metal sources",
                      t.get("manifest"));
      } else
        schema.fail("VALUE", where + ".source must be a .metal, .prx or .prk kernel module",
                    t.get("source"));
      out.modules[name] = m;
    }
  if (auto cases = root.get("cases"))
    for (auto &&[key, node] : schema.table(*cases, "cases")) {
      const std::string name(key.str()), where = "cases." + name;
      auto &t = schema.table(node, where);
      schema.only(t, {"module", "file"}, where);
      ProjectCase c;
      c.name = name;
      c.module = schema.text(schema.need(t, "module", where), where + ".module");
      if (!out.modules.count(c.module))
        schema.fail("REFERENCE", where + ".module names no declared module", t.get("module"));
      c.file = resolve(base, schema.text(schema.need(t, "file", where), where + ".file"));
      out.cases[name] = c;
    }
  if (auto programs = root.get("programs"))
    for (auto &&[key, node] : schema.table(*programs, "programs")) {
      const std::string name(key.str()), where = "programs." + name;
      auto &t = schema.table(node, where);
      schema.only(t, {"source", "arguments"}, where);
      if (out.cases.count(name))
        schema.fail("AMBIGUOUS-NAME", name + " names both a case and a program", &node);
      ProjectProgram p;
      p.name = name;
      p.source = resolve(base, schema.text(schema.need(t, "source", where), where + ".source"));
      if (auto arguments = t.get("arguments")) {
        if (!arguments->is_array())
          schema.fail("TYPE", where + ".arguments must be an array of strings", arguments);
        for (auto &a : *arguments->as_array()) {
          if (!a.is_string())
            schema.fail("TYPE", where + ".arguments must contain only strings", &a);
          p.arguments.push_back(a.as_string()->get());
        }
      }
      out.programs[name] = p;
    }
  if (out.cases.empty() && out.programs.empty())
    schema.fail("EMPTY", "declare at least one [cases.NAME] or [programs.NAME] target", &root);
  if (!out.default_target.empty() && !out.cases.count(out.default_target) &&
      !out.programs.count(out.default_target))
    schema.fail("REFERENCE", "project.default names no declared case or program",
                info.get("default"));
  return out;
}

ProjectSelection select_target(const Project &project, const std::string &case_name,
                               const std::string &program_name) {
  ProjectSelection s;
  auto names = [&] {
    std::string text;
    for (auto &[n, c] : project.cases)
      text += (text.empty() ? "" : ", ") + ("case " + n);
    for (auto &[n, p] : project.programs)
      text += (text.empty() ? "" : ", ") + ("program " + n);
    return text;
  };
  if (!case_name.empty() && !program_name.empty())
    throw Diagnostic("P-PROJECT-SELECTION-AMBIGUOUS", "input",
                     "Select either --case or --program, not both");
  std::string name = !case_name.empty() ? case_name : program_name;
  if (name.empty()) {
    if (!project.default_target.empty())
      name = project.default_target;
    else if (project.cases.size() + project.programs.size() == 1)
      name = project.cases.empty() ? project.programs.begin()->first
                                   : project.cases.begin()->first;
    else
      throw Diagnostic("P-PROJECT-SELECTION-AMBIGUOUS", "input",
                       project.path.string() + " declares several targets and no project.default; "
                       "select one with --case NAME or --program NAME (" + names() + ")");
  }
  bool as_case = case_name.empty() && program_name.empty() ? project.cases.count(name) > 0
                                                           : !case_name.empty();
  if (as_case) {
    auto it = project.cases.find(name);
    if (it == project.cases.end())
      throw Diagnostic("P-PROJECT-SELECTION-UNKNOWN", "input",
                       "No case named " + name + " in " + project.path.string() + " (" + names() +
                           ")");
    s.kernel_case = it->second;
    s.module = &project.modules.at(it->second.module);
  } else {
    auto it = project.programs.find(name);
    if (it == project.programs.end())
      throw Diagnostic("P-PROJECT-SELECTION-UNKNOWN", "input",
                       "No program named " + name + " in " + project.path.string() + " (" +
                           names() + ")");
    s.program = it->second;
  }
  return s;
}
} // namespace paralyn::cli
