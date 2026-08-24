"""Conservative static audit for repository model-deserialization entry points.

The analyzer is deliberately order- and control-flow-independent. It collects
ordinary imports and syntactic assignments from the whole AST, propagates
canonical aliases to a fixed point, then audits references and calls. Dynamic
``getattr`` on a known serialization module fails closed. This covers common
Python source patterns, not arbitrary ``eval`` or runtime metaprogramming.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

KNOWN_MODULES = {
    "torch",
    "torch.serialization",
    "torch.jit",
    "torch.package",
    "torch.hub",
    "pickle",
    "joblib",
}
DANGEROUS_SYMBOLS = {
    "torch.load",
    "torch.serialization.load",
    "torch.jit.load",
    "torch.package.PackageImporter",
    "torch.hub.load_state_dict_from_url",
    "pickle.load",
    "pickle.loads",
    "pickle.Unpickler",
    "joblib.load",
}


@dataclass(frozen=True)
class DeserializationCall:
    path: str
    function: str
    qualified_name: str
    line: int


def _key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _key(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _targets(target: ast.expr, value: ast.expr) -> list[tuple[str, ast.expr]]:
    key = _key(target)
    if key is not None:
        return [(key, value)]
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        return [
            pair
            for target_item, value_item in zip(target.elts, value.elts)
            for pair in _targets(target_item, value_item)
        ]
    return []


def _resolve(
    node: ast.expr, symbols: dict[str, set[str]]
) -> set[str]:
    if isinstance(node, ast.Name):
        return set(symbols.get(node.id, ()))
    if isinstance(node, ast.Attribute):
        direct = symbols.get(_key(node) or "")
        if direct:
            return set(direct)
        return {f"{base}.{node.attr}" for base in _resolve(node.value, symbols)}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
    ):
        modules = _resolve(node.args[0], symbols) & KNOWN_MODULES
        if not modules:
            return set()
        attribute = node.args[1]
        if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
            return {f"{module}.{attribute.value}" for module in modules}
        return {f"{module}.<dynamic-getattr>" for module in modules}
    return set()


def _collect_bindings(
    tree: ast.Module,
) -> tuple[
    dict[str, set[str]],
    list[tuple[str, ast.expr]],
    list[tuple[str, int]],
]:
    symbols: dict[str, set[str]] = {}
    assignments: list[tuple[str, ast.expr]] = []
    wildcard_imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                name = imported.asname or imported.name.split(".")[0]
                canonical = imported.name if imported.asname else name
                symbols.setdefault(name, set()).add(canonical)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*" and node.module in KNOWN_MODULES:
                    wildcard_imports.append((node.module, node.lineno))
                    continue
                name = imported.asname or imported.name
                symbols.setdefault(name, set()).add(
                    f"{node.module}.{imported.name}"
                )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assignments.extend(_targets(target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignments.extend(_targets(node.target, node.value))
        elif isinstance(node, ast.NamedExpr):
            assignments.extend(_targets(node.target, node.value))
    return symbols, assignments, wildcard_imports


def _propagate_aliases(
    symbols: dict[str, set[str]], assignments: list[tuple[str, ast.expr]]
) -> None:
    changed = True
    while changed:
        changed = False
        for target, value in assignments:
            resolved = _resolve(value, symbols)
            known = symbols.setdefault(target, set())
            if not resolved.issubset(known):
                known.update(resolved)
                changed = True


class _AuditVisitor(ast.NodeVisitor):
    def __init__(
        self, path: str, symbols: dict[str, set[str]]
    ) -> None:
        self.path = path
        self.symbols = symbols
        self.functions = ["<module>"]
        self.findings: list[DeserializationCall] = []

    def _record(self, node: ast.AST, resolved: set[str]) -> None:
        for qualified in sorted(resolved):
            if qualified in DANGEROUS_SYMBOLS or qualified.endswith(
                ".<dynamic-getattr>"
            ):
                self.findings.append(
                    DeserializationCall(
                        self.path,
                        self.functions[-1],
                        qualified,
                        node.lineno,
                    )
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        self.functions.append(node.name)
        for child in node.body:
            self.visit(child)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        resolved = _resolve(node.func, self.symbols)
        self._record(node, resolved)
        # The resolved callable is already recorded; visit its arguments for
        # nested dangerous references without duplicating the callable node.
        if not any(
            value in DANGEROUS_SYMBOLS
            or value.endswith(".<dynamic-getattr>")
            for value in resolved
        ):
            self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Assignment targets may themselves have a propagated dangerous value;
        # audit only the reference being assigned, not the write target.
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._record(node, _resolve(node, self.symbols))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._record(node, _resolve(node, self.symbols))


def find_unsafe_deserialization_calls(
    source: str, path: str
) -> list[DeserializationCall]:
    tree = ast.parse(source, path)
    symbols, assignments, wildcard_imports = _collect_bindings(tree)
    _propagate_aliases(symbols, assignments)
    visitor = _AuditVisitor(path, symbols)
    visitor.visit(tree)
    wildcard_findings = [
        DeserializationCall(
            path,
            "<module>",
            f"{module}.<wildcard-import>",
            line,
        )
        for module, line in wildcard_imports
    ]
    return wildcard_findings + visitor.findings
