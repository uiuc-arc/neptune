#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>

#include <regex>
#include <variant>

namespace tvm::tir {

/*! \brief A simple printer that converts a PrimExpr into a preorder expression (S-expr)
 * string. S-expr is a simple format that is less verbose than TVM's serialization format, and is
 * easier to parse. */
struct SExprPrinter : public ExprFunctor<void(const PrimExpr&)> {
 public:
  static std::string Print(const PrimExpr& expr) {
    SExprPrinter printer;
    printer(expr);
    return printer.ss.str();
  }

 private:
  void VisitExpr_(const VarNode* op) override {
    // NOTE: TVM doesn't mind about variables with the same name, because it distinguishes them by
    // pointer address, but we do need unique names before going to string.
    ss << this->CheckVarNameDups(op) << ":" << op->dtype;
  }

  void VisitExpr_(const IntImmNode* op) override {
    if (op->dtype.is_bool()) {
      ss << (op->value ? "true" : "false");
    } else {
      ss << op->value << ":" << op->dtype;
    }
  }

#define DefineVisitUOp(Type, Op)                   \
  void VisitExpr_(const Type##Node* op) override { \
    ss << "(" << Op << " ";                        \
    VisitExpr(op->a);                              \
    ss << ")";                                     \
  }
#define DefineVisitBinOp(Type, Op)                 \
  void VisitExpr_(const Type##Node* op) override { \
    ss << "(" << Op << " ";                        \
    VisitExpr(op->a);                              \
    ss << " ";                                     \
    VisitExpr(op->b);                              \
    ss << ")";                                     \
  }
  DefineVisitBinOp(Add, "+");
  DefineVisitBinOp(Sub, "-");
  DefineVisitBinOp(Mul, "*");
  // Div for integers is truncdiv(a, b) and
  // equivalent to floordiv(a, b) when both are positive.
  DefineVisitBinOp(Div, "/");
  DefineVisitBinOp(FloorDiv, "//");
  DefineVisitBinOp(FloorMod, "mod");
  DefineVisitBinOp(Min, "min");
  DefineVisitBinOp(Max, "max");
  DefineVisitBinOp(EQ, "==");
  DefineVisitBinOp(NE, "!=");
  DefineVisitBinOp(LT, "<");
  DefineVisitBinOp(LE, "<=");
  DefineVisitBinOp(GT, ">");
  DefineVisitBinOp(GE, ">=");
  DefineVisitBinOp(And, "&&");
  DefineVisitBinOp(Or, "||");
  DefineVisitUOp(Not, "!");
#undef DefineVisitUOp
#undef DefineVisitBinOp

  void VisitExpr_(const CallNode* op) override {
    if (auto* op_node = op->op.as<OpNode>()) {
      ss << "(" << op_node->name << " ";
      bool has_last = false;
      for (const auto& arg : op->args) {
        if (has_last) {
          ss << " ";
        } else {
          has_last = true;
        }
        VisitExpr(arg);
      }
      ss << ")";
    } else {
      LOG(FATAL) << "Only calls to builtin operators are supported";
    }
  }

  void VisitExpr_(const CastNode* op) override {
    ss << "(cast \"" << op->dtype << "\" ";
    VisitExpr(op->value);
    ss << ")";
  }

  void VisitExprDefault_(const Object* op) override {
    // If we get to this point, there's something we don't know how to print.
    LOG_FATAL << "No default for " << op->GetTypeKey() << " in SExprConverter";
  }

  std::string CheckVarNameDups(const VarNode* var) {
    auto it = var_map.find(var->name_hint);
    ICHECK(it == var_map.end() || it->second.get() == var)
        << "A different variable with the same name " << var->name_hint << " already exists";
    return var->name_hint;
  }

  std::ostringstream ss;
  std::unordered_map<std::string, Var> var_map;
};

struct SExprParser {
 public:
  static PrimExpr Parse(const std::string& str,
                        const std::unordered_map<std::string, Var>& var_map) {
    return SExprParser::Parse(str, [&var_map](const std::string& name) {
      auto it = var_map.find(name);
      ICHECK(it != var_map.end()) << "Variable " << name << " not found";
      return it->second;
    });
  }

  // Parse an S expression.
  static PrimExpr Parse(const std::string& str,
                        const std::function<Var(const std::string&)>& subst_f) {
    SExprParser parser(subst_f);
    return parser.ParseInternal(str, /*loc=*/0).first;
  }

 private:
  const std::function<Var(const std::string&)>& subst_f_;

  SExprParser(const std::function<Var(const std::string&)>& subst_f) : subst_f_(subst_f) {}

  PrimExpr ParseTokenAsExpr(const std::string& str) {
    ICHECK(!str.empty());
    if (str == "true") {
      return Bool(true);
    } else if (str == "false") {
      return Bool(false);
    } 
    static const std::regex main_pat("^(-?[\\w.]+):(\\w+)$"), int_pat("^-?\\d+$");
    std::smatch match;
    ICHECK(std::regex_match(str, match, main_pat)) << "Invalid token " << str;
    DataType dtype(runtime::String2DLDataType(match[2]));
    auto content = match[1].str();
    if (std::regex_match(content, int_pat)) {
      return IntImm(dtype, std::stoll(content));
    }
    Var var = subst_f_(content);
    ICHECK(var->dtype == dtype) << "Expected " << dtype << " from variable, got " << var->dtype;
    return var;
  }

  using ExprOrDType = std::variant<PrimExpr, DataType>;
  ExprOrDType ParseToken(const std::string& str) {
    if (str.front() == '"' && str.back() == '"') {
      return DataType(runtime::String2DLDataType(str.substr(1, str.size() - 2)));
    }
    return ParseTokenAsExpr(str);
  }

  std::pair<PrimExpr, size_t> ParseInternal(const std::string& str, size_t loc) {
    if (str[loc] != '(') {
      if (loc == 0) {
        return {ParseTokenAsExpr(str), loc};
      }
      LOG(FATAL) << "Expected operator at position " << loc << ": " << str;
    }
    std::string op_str;
    std::vector<ExprOrDType> args;
    size_t last_sep = loc;
    ++loc;
    for (; loc < str.size(); ++loc) {
      if (str[loc] == '(') {
        if (op_str.empty()) {
          LOG(FATAL) << "Expected operator at position " << loc << ": " << str;
        }
        auto [subexpr, new_loc] = ParseInternal(str, loc);
        args.push_back(std::move(subexpr));
        loc = new_loc;  // Will be inc'd by loop
        last_sep = new_loc;
      } else if (str[loc] == ' ' || str[loc] == ')') {
        if (last_sep < loc - 1) {
          std::string token = str.substr(last_sep + 1, loc - last_sep - 1);
          if (op_str.empty()) {
            op_str = std::move(token);
          } else {
            args.push_back(ParseToken(token));
          }
        }
        last_sep = loc;
        if (str[loc] == ')') {
          break;
        }
      }
    }
    if (str[loc] != ')') {
      throw std::runtime_error("Expected ')' at end of expression");
    }
    if (op_str.empty()) {
      throw std::runtime_error("Expected operator at position " + std::to_string(loc) + ": " + str);
    }

#define MakeUnOp(op_name, func, args, loc) \
  auto [arg0] = CheckArgsForParsing<1>(args, op_name);   \
  return {func(std::move(arg0)), loc};

#define MakeBinOp(op_name, func, args, loc) \
  auto [arg0, arg1] = CheckArgsForParsing<2>(args, op_name); \
  return {func(std::move(arg0), std::move(arg1)), loc};

    if (op_str == "+") {
      MakeBinOp("Add", add, args, loc);
    } else if (op_str == "-") {
      MakeBinOp("Sub", sub, args, loc);
    } else if (op_str == "*") {
      MakeBinOp("Mul", mul, args, loc);
    } else if (op_str == "/") {
      MakeBinOp("Div", div, args, loc);
    } else if (op_str == "//") {
      MakeBinOp("FloorDiv", floordiv, args, loc);
    } else if (op_str == "mod") {
      MakeBinOp("FloorMod", floormod, args, loc);
    } else if (op_str == "min") {
      MakeBinOp("Min", min, args, loc);
    } else if (op_str == "max") {
      MakeBinOp("Max", max, args, loc);
    } else if (op_str == "==") {
      MakeBinOp("EQ", equal, args, loc);
    } else if (op_str == "!=") {
      MakeBinOp("NE", not_equal, args, loc);
    } else if (op_str == "<") {
      MakeBinOp("LT", less, args, loc);
    } else if (op_str == "<=") {
      MakeBinOp("LE", less_equal, args, loc);
    } else if (op_str == ">") {
      MakeBinOp("GT", greater, args, loc);
    } else if (op_str == ">=") {
      MakeBinOp("GE", greater_equal, args, loc);
    } else if (op_str == "&&") {
      MakeBinOp("And", logical_and, args, loc);
    } else if (op_str == "||") {
      MakeBinOp("Or", logical_or, args, loc);
    } else if (op_str == "!") {
      MakeUnOp("Not", logical_not, args, loc);
    } else if (op_str == "tir.exp") {
      MakeUnOp("Exp", exp, args, loc);
    } else if (op_str == "cast") {
      auto arg0 = std::get<DataType>(args[0]);
      auto arg1 = std::get<PrimExpr>(args[1]);
      return {cast(arg0, std::move(arg1)), loc};
    }
    LOG_FATAL << "Unknown operator " << op_str << " in " << str;
  }

  template <size_t N>
  std::array<PrimExpr, N> CheckArgsForParsing(std::vector<ExprOrDType>& args, const std::string& op_name) {
    if (args.size() != N) {
      std::ostringstream ss;
      ss << "Expected " << N << " arguments for " << op_name << ", got " << args.size() << " arguments";
      throw std::runtime_error(ss.str());
    }
    std::array<PrimExpr, N> result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = std::get<PrimExpr>(args[i]);
    }
    return result;
  }
};

}  // namespace tvm::tir
