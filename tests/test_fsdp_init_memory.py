# SPDX-License-Identifier: Apache-2.0
"""Guard the one-time cache handoff after full-model loading and FSDP sharding."""

import ast
from pathlib import Path


def test_fsdp_init_releases_full_state_and_cache_after_sharding_and_optimizer():
    """Colocated models must not retain another process's loading cache."""
    source = Path(__file__).resolve().parents[1] / "areal/engine/fsdp_engine.py"
    tree = ast.parse(source.read_text())
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPEngine"
    )
    init = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize"
    )
    calls = {
        ast.unparse(node.value.func): index
        for index, node in enumerate(init.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    release = max(
        index
        for index, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "full_state" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    )
    assert calls["parallelize_model"] < calls["self._create_optimizer"]
    assert calls["self._create_optimizer"] < release
    assert release < calls["current_platform.clear_memory"]
