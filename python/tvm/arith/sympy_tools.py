from collections import OrderedDict
from functools import wraps

import sympy as sp

import tvm
from tvm import tir
from tvm.ir import PrimExpr


def _find_visitor(cls_instance, type: type):
    for class_ in type.mro():
        visitor = getattr(cls_instance, f"visit_{class_.__name__.lower()}", None)
        if visitor is not None:
            return visitor
    return None


class cast(sp.Function):
    nargs = 2  # we have two arguments: the expression and the dtype

    @classmethod
    def eval(cls, expr, dtype):
        # Flatten nested casts: cast(cast(x, dtype1), dtype2) -> cast(x, dtype2)
        if expr.func == cls:
            return cls(expr.args[0], dtype)

    def _combine_same_type(self, other, op):
        """
        Helper for arithmetic operations: if two cast expressions have the same dtype,
        combine them using the operation `op` on their underlying expressions.
        """
        if isinstance(other, cast):
            if self.args[1] != other.args[1]:
                raise ValueError("Cannot combine cast expressions with different dtypes.")
            new_expr = op(self.args[0], other.args[0])
            return cast(new_expr, self.args[1])
        else:
            new_expr = op(self.args[0], other)
            return cast(new_expr, self.args[1])

    def __add__(self, other):
        return self._combine_same_type(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._combine_same_type(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._combine_same_type(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._combine_same_type(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._combine_same_type(other, lambda a, b: a / b)


class _PrimExprToSympy:
    def __init__(self, var_to_var: dict[tir.Var, sp.Symbol]) -> None:
        self.var_to_var = var_to_var

    def __call__(self, expr: PrimExpr) -> sp.Expr:
        visitor = _find_visitor(self, type(expr))
        if visitor is None:
            raise ValueError(f"Unsupported TVM expression: {expr} of type {type(expr)}")
        return visitor(expr)

    @staticmethod
    def _visit_binop(sympy_binop):
        return lambda self, expr: sympy_binop(self(expr.a), self(expr.b))

    visit_add = _visit_binop(lambda a, b: a + b)
    visit_sub = _visit_binop(lambda a, b: a - b)
    visit_mul = _visit_binop(lambda a, b: a * b)
    visit_div = _visit_binop(lambda a, b: a / b)
    visit_floordiv = _visit_binop(lambda a, b: a // b)

    def visit_cast(self, expr: tir.Cast):
        return cast(self(expr.value), str(expr.dtype))

    def visit_var(self, expr: tir.Var):
        if expr in self.var_to_var:
            return self.var_to_var[expr]
        assumptions = {}
        if expr.dtype.startswith("float"):
            assumptions["real"] = True
        symbol = sp.Symbol(expr.name, **assumptions)
        self.var_to_var[expr] = symbol
        return symbol

    def visit_call(self, expr: tir.Call):
        op = expr.op
        if not isinstance(op, tir.op.Op):
            raise ValueError(f"Unsupported call function: {op}")
        args = [self(arg) for arg in expr.args]
        return self.OP_DICT[op.name](*args)

    def visit_intimm(self, expr: tir.IntImm):
        return sp.Integer(expr.value)

    def visit_floatimm(self, expr: tir.FloatImm):
        if int(expr.value) == expr.value:
            return sp.Integer(expr.value)
        return sp.Float(expr.value)

    OP_DICT = {
        "tir.exp": sp.exp,
        "tir.log": sp.log,
        "tir.sqrt": sp.sqrt,
    }


class _SympyToPrimExpr:
    def __init__(self, var_to_var: dict[sp.Symbol, tir.Var], global_dtype: str) -> None:
        self.var_to_var = var_to_var
        self.global_dtype = global_dtype

    def __call__(self, expr: sp.Basic) -> tir.PrimExpr:
        visitor = _find_visitor(self, type(expr))
        if visitor is None:
            raise ValueError(f"Unsupported sympy expression: {expr} of type {type(expr)}")
        return visitor(expr)

    @staticmethod
    def _visit_binop(tir_binop):
        return lambda self, expr: tir_binop(self(expr.args[0]), self(expr.args[1]))

    def visit_cast(self, expr: cast):
        # NOTE: sympy automatically converts dtype from a string into a symbol,
        # we're just converting it back.
        return tir.Cast(value=self(expr.args[0]), dtype=str(expr.args[1]))

    def visit_add(self, expr: sp.Add):
        def _detect_neg1_mul(expr):
            if isinstance(expr, sp.Mul):
                if len(expr.args) == 2 and expr.args[0] == -1:
                    return expr.args[1]
            return None

        ret = self(expr.args[0])
        for arg in expr.args[1:]:
            neg1_mul = _detect_neg1_mul(arg)
            if neg1_mul is not None:
                ret = tir.Sub(ret, self(neg1_mul))
            else:
                ret = tir.Add(ret, self(arg))
        return ret

    def visit_mul(self, expr: sp.Mul):
        def _detect_neg1_pow(expr):
            if isinstance(expr, sp.Pow):
                if len(expr.args) == 2 and expr.args[1] == -1:
                    return expr.args[0]
            return None

        ret = self(expr.args[0])
        for arg in expr.args[1:]:
            neg1_pow = _detect_neg1_pow(arg)
            if neg1_pow is not None:
                ret = tir.Div(ret, self(neg1_pow))
            else:
                ret = tir.Mul(ret, self(arg))
        return ret

    def visit_symbol(self, expr: sp.Symbol):
        return self.var_to_var[expr]

    def visit_exp(self, expr: sp.exp):
        return tir.exp(self(expr.args[0]))

    def visit_log(self, expr: sp.log):
        return tir.log(self(expr.args[0]))

    def visit_pow(self, expr: sp.Pow):
        if expr.args[1] == -1:
            return tir.div(1, self(expr.args[0]))
        return tir.pow(self(expr.args[0]), self(expr.args[1]))

    def visit_integer(self, expr: sp.Integer):
        return tir.const(expr, self.global_dtype)

    def visit_float(self, expr: sp.Float):
        return tir.const(expr, self.global_dtype)

    def visit_booleanatom(self, expr):
        if expr is sp.true:
            return tir.const(True, "bool")
        elif expr is sp.false:
            return tir.const(False, "bool")
        raise ValueError(f"Unsupported boolean atom: {expr}")


class TIRSympyConverter:
    def __init__(self) -> None:
        self.var_tir_to_sp: dict[tir.Var, sp.Symbol] = {}

    def to_sympy(self, expr: PrimExpr):
        converter = _PrimExprToSympy(self.var_tir_to_sp)
        return converter(expr)

    def to_tir(self, expr: sp.Basic, global_dtype: str):
        var_sp_to_tir = {v: k for k, v in self.var_tir_to_sp.items()}
        converter = _SympyToPrimExpr(var_sp_to_tir, global_dtype)
        return converter(expr)


def repr_lru_cache(maxsize=128):
    def decorator(func):
        cache = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key based on the string representation of arguments
            key = repr((args, kwargs))
            if key in cache:
                # Move the accessed item to the end to maintain LRU order
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                # Remove the oldest item (LRU policy)
                cache.popitem(last=False)
            return result

        return wrapper

    return decorator


@repr_lru_cache()
def _cached_simplify(expr: sp.Basic):
    return sp.simplify(expr)


@repr_lru_cache()
def _cached_solve(expr: sp.Basic, symbol: sp.Symbol):
    return sp.solve(expr, symbol)


@tvm._ffi.register_func("arith.sympy.separate_vars")
def separate_vars(expr: tir.PrimExpr, vars: list[tir.Var], a: tir.Var, b: tir.Var):
    """Rewrite a function `f(x..., u...)` into a variable-separated form.

    Separate variables in a function `f(x..., u...)` so that
    `f(x..., u...) = f3(f1(x...), f2(u...))`.
    `f3` is an R x R -> R function.
    On success, returns `f1`, `f2` as expressions, and `f3(a, b)` as expression of `a` and `b`.
    Returns None if failed.

    Parameters
    ----------
    expr :
        The expression of the function `f`.
    vars :
        `x`-variables. A list of variables to separate out from the expression.
        Variables that occur in `expr` and not in `vars` are defaulted as
        the other set of variables (`u`-variables).
    a :
        Variable `a` used in creating `f3(a, b)` as an expression.
    b :
        Variable `b` used in creating `f3(a, b)` as an expression.
    """
    converter = TIRSympyConverter()
    sexpr = converter.to_sympy(expr)
    x_vars = set([converter.var_tir_to_sp[v] for v in vars])

    def contains_only_xvars(expr: sp.Basic):
        expr_vars = expr.free_symbols
        if expr_vars.issubset(x_vars):  # only x-vars
            return True
        if not expr_vars.intersection(x_vars):  # only u-vars
            return False
        return None  # mixed

    sexpr: sp.Expr = sp.expand(sexpr)
    if not isinstance(sexpr, (sp.Mul, sp.Add)):
        print(f"Cannot split unsupported expression: {sexpr}")
        return None
    x_terms, u_terms = [], []
    for arg in sexpr.args:
        match contains_only_xvars(arg):
            case True:
                x_terms.append(arg)
            case False:
                u_terms.append(arg)
            case None:
                print(f"Cannot split mixed expression: {arg} in {sexpr}")
                return None
    if isinstance(sexpr, sp.Mul):
        x_expr, u_expr = sp.Mul(*x_terms), sp.Mul(*u_terms)
        f_comb = a * b
    else:  # Add
        x_expr, u_expr = sp.Add(*x_terms), sp.Add(*u_terms)
        f_comb = a + b
    x_expr, u_expr = _cached_simplify(x_expr), _cached_simplify(u_expr)
    x_expr = converter.to_tir(x_expr, expr.dtype)
    u_expr = converter.to_tir(u_expr, expr.dtype)
    return [x_expr, u_expr, f_comb]


@tvm._ffi.register_func("arith.sympy.inverse")
def invert_binfunc(expr: tir.PrimExpr, y: tir.Var, z: tir.Var):
    """Finds the inverse of `y = f(x, ...) = expr` against `y` as a function `f^-1(z, x)`.
    Returns a PrimExpr containing `x` and `z`.

    - If `expr` contains variables other than `x`, they won't be checked for or considered,
      (so they will be present in the output), but they may interfere with equation solving.
    - Cast expressions are unsupported and everything will be in the type expr.dtype.
    """
    converter = TIRSympyConverter()
    sexpr = converter.to_sympy(expr)
    sz = sp.Symbol("z")
    solutions = _cached_solve(sp.Eq(sexpr, sz), converter.var_tir_to_sp[y])
    if not solutions:
        raise ValueError("No solution found")
    elif len(solutions) > 1:
        raise ValueError("Multiple solutions found")
    sol_expr = solutions[0]
    converter.var_tir_to_sp[z] = sz
    return converter.to_tir(sol_expr, expr.dtype)


@tvm._ffi.register_func("arith.sympy.simplify")
def sympy_simplify(expr: PrimExpr) -> PrimExpr:
    converter = TIRSympyConverter()
    sexpr = converter.to_sympy(expr)
    simpl = _cached_simplify(sexpr)
    return converter.to_tir(simpl, expr.dtype)


@tvm._ffi.register_func("arith.sympy.prove")
def sympy_prove(lhs: PrimExpr, rhs: PrimExpr, cmp: str) -> PrimExpr:
    converter = TIRSympyConverter()
    lhs_, rhs_ = converter.to_sympy(lhs), converter.to_sympy(rhs)
    CMP = {
        "eq": sp.Eq,
        "ne": sp.Ne,
        "lt": sp.Lt,
        "le": sp.Le,
        "gt": sp.Gt,
        "ge": sp.Ge,
    }
    prepos = CMP[cmp](lhs_, rhs_)
    return converter.to_tir(_cached_simplify(prepos), "bool")
