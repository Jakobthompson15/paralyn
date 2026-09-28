#include <cstdlib>
#include "paralyn/frontend.hpp"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/Attr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/SmallString.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace paralyn {
namespace {
using clang::SourceLocation;
using clang::SourceRange;

const char *context_sensitive_feature(llvm::StringRef name) {
  if (name == "__CUDA_ARCH__")
    return "device_architecture_preprocessing";
  if (name == "__CUDA__" || name == "__CUDACC__")
    return "cuda_conditioned_preprocessing";
  if (name == "__LINE__" || name == "__FILE__" || name == "__BASE_FILE__" ||
      name == "__FILE_NAME__" || name == "__COUNTER__" || name == "__INCLUDE_LEVEL__" ||
      name == "__func__" || name == "__FUNCTION__" || name == "__PRETTY_FUNCTION__" ||
      name == "__builtin_LINE" || name == "__builtin_COLUMN" || name == "__builtin_FILE" ||
      name == "__builtin_FILE_NAME" || name == "__builtin_FUNCTION")
    return "source_location_or_context_expression";
  return nullptr;
}

[[noreturn]] void unsupported(const clang::SourceManager &sm, SourceLocation location,
                              const std::string &feature) {
  auto p = sm.getPresumedLoc(sm.getExpansionLoc(location));
  std::ostringstream out;
  if (p.isValid())
    out << p.getFilename() << ':' << p.getLine() << ':' << p.getColumn() << ": ";
  out << "ParalynError: CUDA feature \"" << feature << "\" is not currently supported.";
  throw std::runtime_error(out.str());
}

// Raw Clang tokens include inactive preprocessor branches, but exclude comments
// and string contents. A host-only parse must never silently select a different
// device implementation guarded by __CUDA_ARCH__.
class PreprocessorGuard final : public clang::PPCallbacks {
public:
  explicit PreprocessorGuard(clang::Preprocessor &pp) : pp_(pp) {}
  void FileChanged(SourceLocation loc, FileChangeReason reason,
                   clang::SrcMgr::CharacteristicKind type,
                   clang::FileID = clang::FileID()) override {
    if (reason != EnterFile || type != clang::SrcMgr::C_User)
      return;
    auto &sm = pp_.getSourceManager();
    const auto id = sm.getFileID(loc);
    if (id.isInvalid() || !sm.getFileEntryRefForID(id) || !seen_.insert(id.getHashValue()).second)
      return;
    auto buffer = sm.getBufferOrNone(id);
    if (!buffer)
      return;
    clang::Lexer lexer(id, *buffer, sm, pp_.getLangOpts());
    clang::Token token;
    do {
      lexer.LexFromRawLexer(token);
      if (token.is(clang::tok::raw_identifier))
        if (const auto *feature = context_sensitive_feature(token.getRawIdentifier()))
          unsupported(sm, token.getLocation(), feature);
    } while (!token.is(clang::tok::eof));
  }
  void MacroExpands(const clang::Token &token, const clang::MacroDefinition &, SourceRange,
                    const clang::MacroArgs *) override {
    auto &sm = pp_.getSourceManager();
    if (sm.isInSystemHeader(sm.getExpansionLoc(token.getLocation())))
      return;
    if (const auto *identifier = token.getIdentifierInfo())
      if (const auto *feature = context_sensitive_feature(identifier->getName()))
        unsupported(sm, token.getLocation(), feature);
  }

private:
  clang::Preprocessor &pp_;
  std::set<unsigned> seen_;
};

struct Edit {
  unsigned begin;
  unsigned end;
  std::string replacement;
};

class Visitor final : public clang::RecursiveASTVisitor<Visitor> {
public:
  Visitor(clang::ASTContext &ctx, FrontendResult &result, std::string source)
      : ctx_(ctx), sm_(ctx.getSourceManager()), result_(result), source_(std::move(source)) {}

  bool VisitFunctionDecl(clang::FunctionDecl *decl) {
    if (!in_main(decl->getLocation()))
      return true;
    if (decl->hasAttr<clang::CUDADeviceAttr>())
      unsupported(sm_, decl->getLocation(), "device_functions");
    if (!decl->hasAttr<clang::CUDAGlobalAttr>())
      return true;
    if (!decl->doesThisDeclarationHaveABody())
      unsupported(sm_, decl->getLocation(), "separate_kernel_declarations");
    if (!decl->getDeclContext()->isTranslationUnit() || decl->isTemplated() || decl->isVariadic() ||
        !decl->getReturnType()->isVoidType())
      unsupported(sm_, decl->getLocation(), "non_plain_kernel_definition");
    for (const auto *attr : decl->attrs())
      if (!llvm::isa<clang::CUDAGlobalAttr>(attr))
        unsupported(sm_, attr->getLocation(), "kernel_attributes");
    const auto name = decl->getNameAsString();
    if (kernels_.count(name))
      unsupported(sm_, decl->getLocation(), "overloaded_kernels");
    names_.clear();
    bindings_.clear();
    Kernel kernel;
    kernel.name = name;
    for (const auto *param : decl->parameters()) {
      if (param->getName().empty())
        unsupported(sm_, param->getLocation(), "unnamed_kernel_parameter");
      Parameter p;
      p.name = param->getNameAsString();
      clang::QualType type = param->getType();
      p.buffer = type->isPointerType();
      if (p.buffer) {
        type = type->getPointeeType();
        p.read_only = type.isConstQualified();
      }
      p.type = scalar_type(type, param->getLocation());
      if (p.type == ScalarType::Bool)
        unsupported(sm_, param->getLocation(), "bool_kernel_parameter");
      names_.insert(p.name);
      bindings_[param] = p;
      kernel.parameters.push_back(p);
    }
    lower_statement(decl->getBody(), kernel.body);
    verify(kernel);
    kernels_[name] = result_.kernels.size();
    result_.kernels.push_back(kernel);
    auto range = file_range(decl->getSourceRange(), false);
    edits_.push_back({range.first, range.second,
                      "static const paralyn::Kernel& __paralyn_generated_kernel_" + name +
                          "() {\n"
                          "  static const paralyn::Kernel kernel = " +
                          emit_cpp(kernel) +
                          ";\n"
                          "  return kernel;\n}\n"});
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *decl) {
    if (in_main(decl->getLocation()) &&
        (decl->hasAttr<clang::CUDADeviceAttr>() || decl->hasAttr<clang::CUDAConstantAttr>() ||
         decl->hasAttr<clang::CUDASharedAttr>()))
      unsupported(sm_, decl->getLocation(), "device_or_shared_storage");
    return true;
  }

  bool VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr *call) {
    if (!in_main(call->getBeginLoc()))
      return true;
    launches_.push_back(call);
    return true;
  }

  bool VisitPredefinedExpr(clang::PredefinedExpr *expression) {
    if (in_main(expression->getExprLoc()))
      unsupported(sm_, expression->getExprLoc(), "source_location_or_context_expression");
    return true;
  }

  bool VisitSourceLocExpr(clang::SourceLocExpr *expression) {
    if (in_main(expression->getExprLoc()))
      unsupported(sm_, expression->getExprLoc(), "source_location_or_context_expression");
    return true;
  }

  void finish() {
    for (auto *call : launches_)
      rewrite_launch(call);
    std::sort(edits_.begin(), edits_.end(),
              [](const Edit &a, const Edit &b) { return a.begin > b.begin; });
    unsigned previous = static_cast<unsigned>(source_.size());
    for (const auto &edit : edits_) {
      if (edit.end > previous || edit.end < edit.begin)
        throw std::runtime_error("ParalynError: overlapping CUDA source edits.");
      source_.replace(edit.begin, edit.end - edit.begin, edit.replacement);
      previous = edit.begin;
    }
    result_.rewritten_host =
        "#include <paralyn/runtime.hpp>\n#include <cuda_runtime.h>\n" + source_;
  }

private:
  bool in_main(SourceLocation loc) const {
    return sm_.isWrittenInMainFile(sm_.getExpansionLoc(loc));
  }

  ScalarType scalar_type(clang::QualType type, SourceLocation loc) const {
    if (type.isVolatileQualified())
      unsupported(sm_, loc, "volatile_device_type");
    const auto *builtin = type.getCanonicalType().getUnqualifiedType()->getAs<clang::BuiltinType>();
    if (!builtin)
      unsupported(sm_, loc, "non_scalar_device_type");
    switch (builtin->getKind()) {
    case clang::BuiltinType::Int:
      return ScalarType::I32;
    case clang::BuiltinType::UInt:
      return ScalarType::U32;
    case clang::BuiltinType::Float:
      return ScalarType::F32;
    case clang::BuiltinType::Bool:
      return ScalarType::Bool;
    default:
      unsupported(sm_, loc, "device_type_" + type.getAsString());
    }
  }

  Expr location(Expr expression, SourceLocation loc) const {
    const auto p = sm_.getPresumedLoc(sm_.getExpansionLoc(loc));
    if (p.isValid()) {
      expression.line = p.getLine();
      expression.column = p.getColumn();
    }
    return expression;
  }

  Expr lower_expression(const clang::Expr *input) {
    if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(input))
      return lower_expression(paren->getSubExpr());
    if (const auto *cast = llvm::dyn_cast<clang::CastExpr>(input)) {
      if (cast->getCastKind() == clang::CK_LValueToRValue || cast->getCastKind() == clang::CK_NoOp)
        return lower_expression(cast->getSubExpr());
      if (cast->getCastKind() != clang::CK_IntegralCast)
        unsupported(sm_, input->getExprLoc(),
                    "device_cast_" + std::string(cast->getCastKindName()));
      const auto target = scalar_type(cast->getType(), input->getExprLoc());
      auto value = lower_expression(cast->getSubExpr());
      if ((target != ScalarType::I32 && target != ScalarType::U32) ||
          (value.type != ScalarType::I32 && value.type != ScalarType::U32))
        unsupported(sm_, input->getExprLoc(), "non_integer_device_cast");
      return location({ExprKind::Cast, target, {}, {value}}, input->getExprLoc());
    }
    if (const auto *integer = llvm::dyn_cast<clang::IntegerLiteral>(input)) {
      llvm::SmallString<32> value;
      integer->getValue().toString(value, 10, false);
      return location({ExprKind::Literal,
                       scalar_type(input->getType(), input->getExprLoc()),
                       value.str().str(),
                       {}},
                      input->getExprLoc());
    }
    if (const auto *number = llvm::dyn_cast<clang::FloatingLiteral>(input)) {
      const auto type = scalar_type(input->getType(), input->getExprLoc());
      llvm::SmallString<32> value;
      number->getValue().toString(value, 0, 0);
      return location({ExprKind::Literal, type, value.str().str(), {}}, input->getExprLoc());
    }
    if (const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(input)) {
      auto it = bindings_.find(ref->getDecl());
      if (it == bindings_.end())
        unsupported(sm_, input->getExprLoc(), "nonlocal_device_reference");
      if (it->second.buffer)
        unsupported(sm_, input->getExprLoc(), "device_pointer_expression");
      return location({ExprKind::Ref, it->second.type, it->second.name, {}}, input->getExprLoc());
    }
    if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(input)) {
      const auto *base =
          llvm::dyn_cast<clang::DeclRefExpr>(member->getBase()->IgnoreParenImpCasts());
      const auto name = base ? base->getDecl()->getNameAsString() : "";
      const auto field = member->getMemberNameInfo().getAsString();
      if ((name != "threadIdx" && name != "blockIdx" && name != "blockDim" && name != "gridDim") ||
          (field != "x" && field != "y" && field != "z") || member->isArrow() ||
          !base->getDecl()->hasAttr<clang::CUDADeviceAttr>() ||
          in_main(base->getDecl()->getLocation()))
        unsupported(sm_, input->getExprLoc(), "device_member_access");
      return location({ExprKind::Builtin, ScalarType::U32, name + "." + field, {}},
                      input->getExprLoc());
    }
    if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(input)) {
      if (binary->getOpcode() != clang::BO_Add && binary->getOpcode() != clang::BO_Mul &&
          binary->getOpcode() != clang::BO_LT)
        unsupported(sm_, input->getExprLoc(), "operator_" + binary->getOpcodeStr().str());
      return location({ExprKind::Binary,
                       scalar_type(input->getType(), input->getExprLoc()),
                       binary->getOpcodeStr().str(),
                       {lower_expression(binary->getLHS()), lower_expression(binary->getRHS())}},
                      input->getExprLoc());
    }
    if (const auto *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(input))
      return lower_subscript(subscript, false);
    unsupported(sm_, input->getExprLoc(), std::string("expression_") + input->getStmtClassName());
  }

  Expr lower_subscript(const clang::ArraySubscriptExpr *subscript, bool writing) {
    const auto *ref =
        llvm::dyn_cast<clang::DeclRefExpr>(subscript->getBase()->IgnoreParenImpCasts());
    const auto it = ref ? bindings_.find(ref->getDecl()) : bindings_.end();
    if (it == bindings_.end() || !it->second.buffer)
      unsupported(sm_, subscript->getExprLoc(), "non_parameter_buffer_indexing");
    if (writing && it->second.read_only)
      unsupported(sm_, subscript->getExprLoc(), "store_to_read_only_buffer");
    auto index = lower_expression(subscript->getIdx());
    if (index.type != ScalarType::I32 && index.type != ScalarType::U32)
      unsupported(sm_, subscript->getExprLoc(), "non_integer_buffer_index");
    auto base = location({ExprKind::Ref, it->second.type, it->second.name, {}}, ref->getExprLoc());
    return location({ExprKind::Load, it->second.type, {}, {base, index}}, subscript->getExprLoc());
  }

  void lower_statement(const clang::Stmt *stmt, std::vector<Statement> &output) {
    if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      for (const auto *child : compound->body())
        lower_statement(child, output);
      return;
    }
    if (const auto *declarations = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      for (const auto *declaration : declarations->decls()) {
        const auto *var = llvm::dyn_cast<clang::VarDecl>(declaration);
        if (var && var->hasAttr<clang::CUDASharedAttr>())
          unsupported(sm_, var->getLocation(), "shared_memory");
        if (!var || !var->hasInit() || var->hasGlobalStorage() || var->hasAttrs())
          unsupported(sm_, declaration->getLocation(), "unsupported_local_declaration");
        Parameter binding;
        binding.name = var->getNameAsString();
        binding.type = scalar_type(var->getType(), var->getLocation());
        if (binding.type == ScalarType::Bool)
          unsupported(sm_, var->getLocation(), "bool_device_local");
        if (!names_.insert(binding.name).second)
          unsupported(sm_, var->getLocation(), "shadowed_device_variable");
        auto initial = lower_expression(var->getInit());
        bindings_[var] = binding;
        Statement result;
        result.kind = StmtKind::Let;
        result.name = binding.name;
        result.type = binding.type;
        result.expression = std::move(initial);
        output.push_back(std::move(result));
      }
      return;
    }
    if (const auto *branch = llvm::dyn_cast<clang::IfStmt>(stmt)) {
      if (branch->getElse() || branch->getInit() || branch->getConditionVariable() ||
          branch->isConstexpr())
        unsupported(sm_, stmt->getBeginLoc(), "if_else_or_initializer");
      Statement result;
      result.kind = StmtKind::If;
      result.expression = lower_expression(branch->getCond());
      lower_statement(branch->getThen(), result.body);
      output.push_back(std::move(result));
      return;
    }
    if (const auto *assignment = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
      const auto *target =
          llvm::dyn_cast<clang::ArraySubscriptExpr>(assignment->getLHS()->IgnoreParens());
      if (assignment->getOpcode() == clang::BO_Assign && target) {
        Statement result;
        result.kind = StmtKind::Store;
        result.target = lower_subscript(target, true);
        result.type = result.target.type;
        result.expression = lower_expression(assignment->getRHS());
        output.push_back(std::move(result));
        return;
      }
    }
    unsupported(sm_, stmt->getBeginLoc(), std::string("statement_") + stmt->getStmtClassName());
  }

  std::pair<unsigned, unsigned> file_range(SourceRange range, bool reject_macro) const {
    if (reject_macro && (range.getBegin().isMacroID() || range.getEnd().isMacroID()))
      unsupported(sm_, range.getBegin(), "macro_generated_launch");
    // The __global__ attribute is conventionally a macro. Expanding its begin
    // location is safe only if the complete declaration ends in the main file.
    if (!reject_macro && range.getEnd().isMacroID())
      unsupported(sm_, range.getBegin(), "macro_generated_kernel");
    const auto begin = sm_.getExpansionLoc(range.getBegin());
    const auto end = clang::Lexer::getLocForEndOfToken(sm_.getExpansionLoc(range.getEnd()), 0, sm_,
                                                       ctx_.getLangOpts());
    if (!sm_.isWrittenInMainFile(begin) || !sm_.isWrittenInMainFile(end) || end.isInvalid())
      unsupported(sm_, range.getBegin(), "cross_file_cuda_definition");
    return {sm_.getFileOffset(begin), sm_.getFileOffset(end)};
  }

  std::string source_expression(const clang::Expr *expression) const {
    if (const auto *default_arg = llvm::dyn_cast<clang::CXXDefaultArgExpr>(expression)) {
      (void)default_arg;
      return "0";
    }
    auto range = clang::Lexer::makeFileCharRange(
        clang::CharSourceRange::getTokenRange(expression->getSourceRange()), sm_,
        ctx_.getLangOpts());
    if (range.isInvalid())
      unsupported(sm_, expression->getExprLoc(), "macro_argument_source_range");
    return clang::Lexer::getSourceText(range, sm_, ctx_.getLangOpts()).str();
  }

  static std::string argument_type(const Parameter &p) {
    std::string type;
    switch (p.type) {
    case ScalarType::I32:
      type = "int";
      break;
    case ScalarType::U32:
      type = "unsigned int";
      break;
    case ScalarType::F32:
      type = "float";
      break;
    case ScalarType::Bool:
      type = "bool";
      break;
    }
    return (p.read_only ? "const " : "") + type + (p.buffer ? "*" : "");
  }

  void rewrite_launch(clang::CUDAKernelCallExpr *call) {
    const auto *callee = call->getDirectCallee();
    if (!callee)
      unsupported(sm_, call->getExprLoc(), "indirect_kernel_launch");
    const auto found = kernels_.find(callee->getNameAsString());
    if (found == kernels_.end())
      unsupported(sm_, call->getExprLoc(), "external_kernel_launch");
    const auto &kernel = result_.kernels[found->second];
    if (call->getNumArgs() != kernel.parameters.size())
      unsupported(sm_, call->getExprLoc(), "kernel_argument_count");
    const auto *config = call->getConfig();
    if (config->getNumArgs() != 4)
      unsupported(sm_, call->getExprLoc(), "launch_configuration");
    auto range = file_range(call->getSourceRange(), true);
    const auto grid = source_expression(config->getArg(0));
    const auto block = source_expression(config->getArg(1));
    const auto shared = source_expression(config->getArg(2));
    const auto stream = source_expression(config->getArg(3));
    clang::Expr::EvalResult shared_value;
    if (!llvm::isa<clang::CXXDefaultArgExpr>(config->getArg(2)) &&
        config->getArg(2)->EvaluateAsInt(shared_value, ctx_) && shared_value.Val.getInt() != 0)
      unsupported(sm_, config->getArg(2)->getExprLoc(), "dynamic_shared_memory");
    clang::Expr::EvalResult stream_value;
    if (!llvm::isa<clang::CXXDefaultArgExpr>(config->getArg(3)) &&
        config->getArg(3)->EvaluateAsRValue(stream_value, ctx_) && stream_value.Val.isLValue() &&
        !stream_value.Val.isNullPointer())
      unsupported(sm_, config->getArg(3)->getExprLoc(), "non_default_stream");
    std::ostringstream text;
    text << "([&]() {\n"
         << "  const dim3 __paralyn_generated_grid = (" << grid << ");\n"
         << "  const dim3 __paralyn_generated_block = (" << block << ");\n"
         << "  const std::size_t __paralyn_generated_shared = (" << shared << ");\n"
         << "  const cudaStream_t __paralyn_generated_stream = (" << stream << ");\n"
         << "  if (!paralyn::validate_launch_configuration(__paralyn_generated_shared, "
            "__paralyn_generated_stream)) return;\n";
    for (unsigned i = 0; i < call->getNumArgs(); ++i)
      text << "  " << argument_type(kernel.parameters[i]) << " __paralyn_generated_argument_" << i
           << " = (" << source_expression(call->getArg(i)) << ");\n";
    text << "  paralyn::launch_checked(__paralyn_generated_kernel_" << kernel.name
         << "(), __paralyn_generated_grid, __paralyn_generated_block, {";
    for (unsigned i = 0; i < call->getNumArgs(); ++i) {
      if (i)
        text << ", ";
      const auto &parameter = kernel.parameters[i];
      const auto method = parameter.buffer                    ? "buffer"
                          : parameter.type == ScalarType::I32 ? "i32"
                          : parameter.type == ScalarType::U32 ? "u32"
                                                              : "f32";
      text << "paralyn::Argument::from_" << method << "(__paralyn_generated_argument_" << i << ')';
    }
    text << "});\n}())";
    edits_.push_back({range.first, range.second, text.str()});
    result_.launches.push_back(
        {kernel.name, grid, block,
         sm_.getPresumedLoc(sm_.getExpansionLoc(call->getExprLoc())).getLine()});
  }

  clang::ASTContext &ctx_;
  clang::SourceManager &sm_;
  FrontendResult &result_;
  std::string source_;
  std::set<std::string> names_;
  std::map<const clang::ValueDecl *, Parameter> bindings_;
  std::map<std::string, std::size_t> kernels_;
  std::vector<clang::CUDAKernelCallExpr *> launches_;
  std::vector<Edit> edits_;
};

class Consumer final : public clang::ASTConsumer {
public:
  Consumer(FrontendResult &result, std::string source)
      : result_(result), source_(std::move(source)) {}
  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    if (ctx.getDiagnostics().hasErrorOccurred())
      return;
    Visitor visitor(ctx, result_, source_);
    visitor.TraverseDecl(ctx.getTranslationUnitDecl());
    visitor.finish();
  }

private:
  FrontendResult &result_;
  std::string source_;
};

class Action final : public clang::ASTFrontendAction {
public:
  Action(FrontendResult &result, std::string source)
      : result_(result), source_(std::move(source)) {}
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &ci,
                                                        llvm::StringRef) override {
    ci.getPreprocessor().addPPCallbacks(std::make_unique<PreprocessorGuard>(ci.getPreprocessor()));
    return std::make_unique<Consumer>(result_, source_);
  }

private:
  FrontendResult &result_;
  std::string source_;
};
} // namespace

FrontendResult compile_source(const std::string &path) {
  const auto absolute = std::filesystem::absolute(path).lexically_normal().string();
  std::ifstream stream(absolute, std::ios::binary);
  if (!stream)
    throw std::runtime_error("ParalynError: cannot read CUDA source: " + absolute);
  const std::string source{std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()};
  FrontendResult result;
  std::vector<std::string> arguments{"-x",
                                           "cuda",
                                           "--cuda-host-only",
                                           "-nocudainc",
                                           "-nocudalib",
                                           "-std=c++17",
                                           "-resource-dir=" PARALYN_RESOURCE_DIR,
                                           std::string("-I") + (std::getenv("PARALYN_INCLUDE_DIR") ? std::getenv("PARALYN_INCLUDE_DIR") : PARALYN_INCLUDE_DIR),
                                           "-include",
                                           "paralyn/cuda_parse.hpp"};
  if (std::string(PARALYN_SDK_PATH).size()) { arguments.push_back("-isysroot"); arguments.push_back(PARALYN_SDK_PATH); }
  if (!clang::tooling::runToolOnCodeWithArgs(std::make_unique<Action>(result, source), source,
                                             arguments, absolute, PARALYN_CLANG_PATH))
    throw std::runtime_error(
        "ParalynError: Clang rejected CUDA source; see source diagnostics above.");
  if (result.kernels.empty())
    throw std::runtime_error("ParalynError: source contains no supported CUDA kernel definitions.");
  return result;
}
} // namespace paralyn
