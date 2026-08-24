from __future__ import annotations

import pytest

from tests.deserialization_ast import find_unsafe_deserialization_calls


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ntorch.load('model.pt')",
        "import torch\ntorch.serialization.load('model.pt')",
        "import torch as t\nt.load('model.pt')",
        "import torch.serialization as s\ns.load('model.pt')",
        "from torch import serialization as s\ns.load('model.pt')",
        "from torch import load as x\nx('model.pt')",
        "from torch.serialization import load\nload('model.pt')",
        "import pickle\npickle.load(stream)",
        "import pickle\npickle.loads(payload)",
        "import pickle as p\np.load(stream)",
        "from pickle import loads as x\nx(payload)",
        "import joblib\njoblib.load('model.joblib')",
        "import joblib as j\nj.load('model.joblib')",
        "from joblib import load\nload('model.joblib')",
        "import torch\nunsafe_loader = torch.load\nx = unsafe_loader\nx('model.pt')",
        "import torch.serialization as s\ngetattr(s, 'load')('model.pt')",
        "import pickle as p\nloader = getattr(p, 'loads')\nloader(payload)",
    ],
)
def test_analyzer_finds_aliased_deserialization_calls(source: str) -> None:
    findings = find_unsafe_deserialization_calls(source, "example.py")

    assert findings
    assert {finding.qualified_name for finding in findings} <= {
        "torch.load",
        "torch.serialization.load",
        "pickle.load",
        "pickle.loads",
        "joblib.load",
    }


@pytest.mark.parametrize(
    "source",
    [
        "import torch as t\ngetattr(t, name)('model.pt')",
        "import torch.serialization as s\ngetattr(s, method)('model.pt')",
        "import pickle as p\ngetattr(p, method)(payload)",
        "import joblib as j\ngetattr(j, method)('model.joblib')",
    ],
)
def test_analyzer_fails_closed_for_dynamic_getattr_on_known_modules(
    source: str,
) -> None:
    findings = find_unsafe_deserialization_calls(source, "example.py")

    assert len(findings) == 1
    assert findings[0].qualified_name.endswith(".<dynamic-getattr>")


def test_analyzer_scans_simple_wrapper_bodies() -> None:
    source = """
import torch as t
unsafe_loader = t.load

def wrapper(path):
    loader = unsafe_loader
    return loader(path)
"""

    findings = find_unsafe_deserialization_calls(source, "example.py")

    assert any(
        finding.function == "wrapper" and finding.qualified_name == "torch.load"
        for finding in findings
    )


def test_analyzer_scans_wrappers_inside_control_flow() -> None:
    source = """
import torch

class Loader:
    def load_model(self, path):
        if path:
            loader = torch.serialization.load
            return loader(path)
"""

    findings = find_unsafe_deserialization_calls(source, "example.py")

    assert findings
    assert {finding.function for finding in findings} == {"load_model"}
    assert {finding.qualified_name for finding in findings} == {
        "torch.serialization.load"
    }


@pytest.mark.parametrize(
    "source",
    [
        "import torch\nif True:\n    loader = torch.load\n    loader('x')",
        "def wrapper(path):\n    return torch.load(path)\nimport torch",
        "import torch\nh.loader = torch.load\nh.loader('x')",
        "import torch\nm = torch\nm.load('x')",
    ],
)
def test_analyzer_is_order_independent_and_propagates_object_aliases(
    source: str,
) -> None:
    findings = find_unsafe_deserialization_calls(source, "example.py")

    assert findings
    assert {finding.qualified_name for finding in findings} == {"torch.load"}


@pytest.mark.parametrize(
    ("module", "later_call"),
    [
        ("torch", "load('x')"),
        ("torch.serialization", "load('x')"),
        ("pickle", "loads(payload)"),
        ("joblib", "load('x')"),
    ],
)
def test_analyzer_fails_closed_on_known_wildcard_imports(
    module: str, later_call: str
) -> None:
    findings = find_unsafe_deserialization_calls(
        f"from {module} import *\n{later_call}", "example.py"
    )

    assert findings
    assert findings[0].qualified_name == f"{module}.<wildcard-import>"


def test_analyzer_allows_unrelated_wildcard_import() -> None:
    source = "from math import *\nvalue = sqrt(4)"

    assert find_unsafe_deserialization_calls(source, "example.py") == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import torch\ntorch.jit.load('x')", "torch.jit.load"),
        ("import torch.jit as tj\ntj.load('x')", "torch.jit.load"),
        ("from torch.jit import load as x\nx('x')", "torch.jit.load"),
        ("import torch\nm = torch.jit\nm.load('x')", "torch.jit.load"),
        ("import torch.jit as tj\ngetattr(tj, 'load')('x')", "torch.jit.load"),
        (
            "import pickle as p\np.Unpickler(stream).load()",
            "pickle.Unpickler",
        ),
        (
            "from pickle import Unpickler as U\nctor = U\nu = ctor(stream)\nu.load()",
            "pickle.Unpickler",
        ),
        (
            "import torch\ntorch.package.PackageImporter('model.pt')",
            "torch.package.PackageImporter",
        ),
        (
            "from torch.package import PackageImporter as P\nP('model.pt')",
            "torch.package.PackageImporter",
        ),
        (
            "import torch\ntorch.hub.load_state_dict_from_url('https://example')",
            "torch.hub.load_state_dict_from_url",
        ),
        (
            "from torch.hub import load_state_dict_from_url as load\nload('https://example')",
            "torch.hub.load_state_dict_from_url",
        ),
    ],
)
def test_analyzer_finds_alternate_deserialization_apis(
    source: str, expected: str
) -> None:
    findings = find_unsafe_deserialization_calls(source, "example.py")

    assert findings
    assert expected in {finding.qualified_name for finding in findings}


@pytest.mark.parametrize("module", ["torch.jit", "torch.package", "torch.hub"])
def test_analyzer_fails_closed_on_alternate_api_wildcards(module: str) -> None:
    findings = find_unsafe_deserialization_calls(
        f"from {module} import *", "example.py"
    )

    assert findings[0].qualified_name == f"{module}.<wildcard-import>"


def test_analyzer_ignores_unrelated_load_methods() -> None:
    source = """
class Config:
    def load(self, path):
        return path

config = Config()
config.load('settings.json')
getattr(config, 'load')('settings.json')
"""

    assert find_unsafe_deserialization_calls(source, "example.py") == []
