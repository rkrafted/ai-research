#!/usr/bin/env python3
"""SWE-smith style bug injection: flip the AST, keep only what the tests catch.

A *mutation* is a small, semantics-changing edit to one source file: a flipped
comparison, a swapped boolean operator, a negated branch test, an off-by-one
constant, a branch swap or a dropped return value. `enumerate_mutations` lists
the edits possible in a source file; `apply_mutation` applies exactly one of
them and returns the rewritten module. Positions are stable: a mutation is
identified by the pre-order index of its AST node, so regenerating the list from
the same source always yields the same mutation.

Stdlib only (ast), so this runs offline.
"""
from __future__ import annotations

import ast

# comparison operator -> the operator that breaks it
COMPARE_FLIP = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
BIN_FLIP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.Mod: ast.FloorDiv,
    ast.FloorDiv: ast.Mod,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
}

OPS = (
    "compare", "boolop", "binop", "negate", "denegate", "swap_branches",
    "drop_call", "swap_args", "const_bool", "const_num", "const_str", "return_none",
)


def enumerate_mutations(src: str) -> list[dict]:
    """All single-edit mutations available in `src`, in stable node order."""
    tree = ast.parse(src)
    out: list[dict] = []
    for site, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare):
            for sub, op in enumerate(node.ops):
                if type(op) in COMPARE_FLIP:
                    out.append({"op": "compare", "site": site, "sub": sub,
                                "line": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.BoolOp):
            out.append({"op": "boolop", "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_FLIP:
            out.append({"op": "binop", "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
        elif isinstance(node, (ast.If, ast.While)):
            is_negated = isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
            out.append({"op": "denegate" if is_negated else "negate", "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
            if isinstance(node, ast.If) and node.body and node.orelse:
                out.append({"op": "swap_branches", "site": site, "sub": 0,
                            "line": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            out.append({"op": "drop_call", "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            out.append({"op": "swap_args", "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.Return) and node.value is not None:
            out.append({"op": "return_none", "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                op = "const_bool"
            elif isinstance(v, (int, float)):
                op = "const_num"
            elif isinstance(v, str) and v:
                op = "const_str"
            else:
                continue
            out.append({"op": op, "site": site, "sub": 0,
                        "line": getattr(node, "lineno", 0)})
    return out


def apply_mutation(src: str, mutation: dict) -> str:
    """Return `src` with exactly the one mutation applied."""
    tree = ast.parse(src)
    nodes = list(ast.walk(tree))
    node = nodes[mutation["site"]]
    op = mutation["op"]
    if op == "compare":
        cur = node.ops[mutation["sub"]]
        node.ops[mutation["sub"]] = COMPARE_FLIP[type(cur)]()
    elif op == "boolop":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif op == "binop":
        node.op = BIN_FLIP[type(node.op)]()
    elif op == "negate":
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    elif op == "denegate":
        node.test = node.test.operand
    elif op == "swap_branches":
        node.body, node.orelse = node.orelse, node.body
    elif op == "const_bool":
        node.value = not node.value
    elif op == "const_num":
        node.value = 1 if node.value == 0 else node.value + 1
    elif op == "const_str":
        node.value = node.value + "_x"
    elif op == "return_none":
        node.value = ast.Constant(value=None)
    elif op == "drop_call":
        node.value = ast.Constant(value=None)
    elif op == "swap_args":
        node.args[0], node.args[1] = node.args[1], node.args[0]
    else:  # pragma: no cover - guarded by OPS
        raise ValueError(f"unknown mutation op: {op}")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def signature(mutation: dict) -> str:
    """Stable, human-readable id for a mutation (leaks no source text)."""
    return f"{mutation['op']}:{mutation['site']}:{mutation['sub']}"
