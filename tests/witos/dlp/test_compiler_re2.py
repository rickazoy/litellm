"""The re2 rule and its rejection path (§2.8, §2.14).

The most important assertion in this file is a negative one: nothing in the
policy fabric may fall back to Python's `re` when re2 refuses a pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import policy_document, scoped

from litellm.proxy.witos.policy_fabric import compiler as compiler_module
from litellm.proxy.witos.policy_fabric.compiler import (
    CompileFailure,
    CompiledPolicy,
    Re2Pattern,
    _MissingRe2,
    compile_policy,
)
from litellm.proxy.witos.policy_fabric.types import CompileStatus

# Constructs re2 refuses by design: backreferences and lookaround are exactly
# what makes a backtracking engine explode on crafted input.
BACKTRACKING_PATTERNS = (
    r"(a+)+\1",
    r"(?=secret)cardholder",
    r"(?<!allow)deny",
    r"(x)\1{100}",
)


def test_re2_rejects_backtracking_constructs_and_the_compile_fails() -> None:
    for pattern in BACKTRACKING_PATTERNS:
        document = policy_document("bad", condition={"type": "regex", "pattern": pattern})
        result = compile_policy(scoped(document).policy)
        assert isinstance(result, CompileFailure), pattern
        assert result.errors[0].pattern == pattern
        assert "re2 rejected" in result.errors[0].reason


def test_a_rejected_pattern_surfaces_for_review_rather_than_being_dropped() -> None:
    document = policy_document(
        "mixed",
        condition={
            "operator": "ANY",
            "conditions": [
                {"type": "regex", "pattern": r"\d{3}-\d{2}-\d{4}"},
                {"type": "regex", "pattern": r"(a+)+\1"},
            ],
        },
    )
    result = compile_policy(scoped(document).policy)
    assert isinstance(result, CompileFailure)
    assert result.summary
    assert "c.1" in result.summary


def test_compiler_never_falls_back_when_re2_is_unavailable() -> None:
    """No re2 means no local detector, not a quietly weaker one."""
    document = policy_document("pci", condition={"type": "regex", "pattern": r"\d{16}"})
    result = compile_policy(scoped(document).policy, re2_module=_MissingRe2())
    assert isinstance(result, CompileFailure)
    assert "google-re2 is not installed" in result.errors[0].reason
    assert "not supported" in result.errors[0].reason


def test_policy_fabric_never_imports_pythons_re_for_matching() -> None:
    """A source-level guard: `re` must not appear anywhere in the package.

    A future edit that adds `import re` to "just handle this one pattern"
    reintroduces the catastrophic-backtracking exposure re2 exists to remove,
    and would do so invisibly. This test makes it visible.
    """
    package = Path(compiler_module.__file__).parent
    offenders = tuple(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() in ("import re", "from re import *") or line.strip().startswith("from re import ")
    )
    assert offenders == (), f"policy fabric must not import Python's re: {offenders}"


def test_terms_leaf_compiles_through_re2_with_word_boundaries() -> None:
    document = policy_document(
        "kw",
        on_match="AUDIT",
        condition={"type": "keyword", "terms": ["merger"], "whole_word": True, "classifier": "IP.TRADE_SECRET"},
    )
    compiled = compile_policy(scoped(document).policy)
    assert isinstance(compiled, CompiledPolicy)
    program: Re2Pattern = compiled.patterns["c"].program
    assert [match.start() for match in program.finditer("the merger is on")] == [4]
    assert [match.start() for match in program.finditer("premerging")] == []


def test_regex_metacharacters_in_terms_are_escaped_not_interpreted() -> None:
    document = policy_document(
        "literal",
        on_match="AUDIT",
        condition={"type": "dictionary", "terms": ["a.c"], "whole_word": False},
    )
    compiled = compile_policy(scoped(document).policy)
    assert isinstance(compiled, CompiledPolicy)
    program = compiled.patterns["c"].program
    assert list(program.finditer("abc")) == []
    assert len(list(program.finditer("a.c"))) == 1


@pytest.mark.parametrize(
    ("condition", "expected"),
    (
        ({"type": "regex", "pattern": r"\d{4}"}, CompileStatus.COMPILED_LOCAL),
        ({"type": "data_class", "class": "PII.SSN"}, CompileStatus.COMPILED_LOCAL),
        ({"type": "data_class", "class": "CUSTOM.cyera.LEARNED_THING"}, CompileStatus.DELEGATED),
        (
            {
                "operator": "ALL",
                "conditions": [
                    {"type": "regex", "pattern": r"\d{4}"},
                    {"type": "data_class", "class": "CUSTOM.cyera.LEARNED_THING"},
                ],
            },
            CompileStatus.HYBRID,
        ),
    ),
)
def test_compile_status_reflects_where_each_leaf_can_be_answered(condition: dict, expected: CompileStatus) -> None:
    compiled = compile_policy(scoped(policy_document("s", condition=condition)).policy)
    assert isinstance(compiled, CompiledPolicy)
    assert compiled.compile_status is expected
