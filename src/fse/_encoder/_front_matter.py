#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The YAML front matter of a model card, read by PyYAML for the few keys fse reads.

ARCHITECTURE.md, "from_pretrained". The block between the first two lines ``---`` (or
``---`` and ``...``) is composed as YAML by ``yaml.SafeLoader`` into a node graph, which
checks the syntax of the whole block. Only the values of the keys fse reads are then
constructed, and only when they are scalars or flat lists of scalars: no other part of the
card is ever constructed, so no tag of an unread key, no timestamp, no merge and no
object of an unread key can run or fail. Aliases are never expanded (they share a node) and
a node that is not a scalar or a scalar list is refused before it is constructed. Every
failure of PyYAML (syntax, a tag it cannot construct, an invalid date, a nesting too deep to
compose, a NUL) is a ``CorruptArtifactError`` that names the file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from fse._container import CorruptArtifactError

_STR = "tag:yaml.org,2002:str"
_NULL = "tag:yaml.org,2002:null"
_MERGE = "tag:yaml.org,2002:merge"
_SEQ = "tag:yaml.org,2002:seq"
_KINDS = ((MappingNode, "mapping"), (SequenceNode, "list"), (ScalarNode, "scalar"))


def front_matter_text(text: str) -> str | None:
    """The lines between the first two lines ``---`` of a card, or None for a card
    without a closed front matter."""
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != "---":
        return None
    for k, line in enumerate(lines[1:], start=1):
        if line.rstrip() in ("---", "..."):
            return "\n".join(lines[1:k])
    return None


def front_matter(data: bytes, path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    """The values of ``keys`` in the front matter of a card: None (no key, or null), a str,
    or a list of the scalars of a flat list. A front matter that is no YAML mapping, merges
    keys, states a key twice or holds a value of a key that is neither a scalar nor a flat
    list raises ``CorruptArtifactError``, as does any YAML error in the block."""
    text = front_matter_text(data.decode("utf-8", errors="strict"))
    found: dict[str, Any] = dict.fromkeys(keys)
    if text is None:
        return found
    loader = None
    try:
        loader = yaml.SafeLoader(text)
        root = loader.get_single_node()
        if root is None or (isinstance(root, ScalarNode) and root.tag == _NULL):
            return found
        if not isinstance(root, MappingNode):
            raise CorruptArtifactError(
                f"{path}: the front matter must be a YAML mapping, not {_kind(root)}"
            )
        for key in keys:
            node = _value_node(root, key, path)
            if node is not None:
                found[key] = _construct(loader, node, key, path)
    except CorruptArtifactError:
        raise
    except Exception as err:  # PyYAML's own errors, a RecursionError, what a constructor raises
        raise CorruptArtifactError(
            f"{path}: front matter fse cannot read as YAML: {_why(err)}"
        ) from err
    finally:
        if loader is not None:
            loader.dispose()
    return found


def _kind(node: Node) -> str:
    return next(name for cls, name in _KINDS if isinstance(node, cls))


def _why(err: BaseException) -> str:
    """What PyYAML says went wrong and the line of the file, where it knows it."""
    if isinstance(err, RecursionError):
        return "nested too deeply"
    problem = getattr(err, "problem", None) or getattr(err, "reason", None) or str(err)
    mark = getattr(err, "problem_mark", None)
    where = "" if mark is None else f" (line {mark.line + 2})"
    return f"{problem}{where}"


def _value_node(root: MappingNode, key: str, path: Path) -> Node | None:
    """The value node of ``key`` among the keys of the root (a quoted key is the same key),
    None if it is absent. A merge key or the key twice raises."""
    found = []
    for key_node, value_node in root.value:
        if not isinstance(key_node, ScalarNode):
            continue
        if key_node.tag == _MERGE:
            raise CorruptArtifactError(
                f"{path}: the front matter merges keys (<<), which fse does not read"
            )
        if key_node.tag == _STR and key_node.value == key:
            found.append(value_node)
    if len(found) > 1:
        raise CorruptArtifactError(f"{path} states {key} {len(found)} times")
    return found[0] if found else None


def _construct(loader: yaml.SafeLoader, node: Node, key: str, path: Path) -> Any:
    """A scalar or a flat list of scalars; any other node is refused unconstructed."""
    if isinstance(node, ScalarNode):
        return _scalar(loader, node)
    flat = isinstance(node, SequenceNode) and all(
        isinstance(item, ScalarNode) for item in node.value
    )
    if flat and node.tag == _SEQ:  # a list of another tag is not constructed
        return [_scalar(loader, item) for item in node.value]
    raise CorruptArtifactError(
        f"{path} states a {key} fse cannot read: a {_kind(node)} that is no scalar or flat list of scalars"
    )


def _scalar(loader: yaml.SafeLoader, node: Node) -> Any:
    """The value of a scalar node; a value its constructor rejects (an invalid date, a
    number or boolean of an explicit tag: a ValueError, a KeyError, an AttributeError) is a
    YAML error at the line of the node."""
    try:
        return loader.construct_object(node, deep=True)
    except yaml.YAMLError:
        raise
    except Exception as err:
        problem = f"{node.tag} is no valid value ({type(err).__name__}: {err})"
        raise yaml.constructor.ConstructorError(None, None, problem, node.start_mark) from err


def strings(value: Any, key: str, path: Path, pattern: re.Pattern[str]) -> str | list[str] | None:
    """The value of a card field as a string or a non-empty list of strings that match
    ``pattern``; None stays None. Another form raises."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            raise CorruptArtifactError(f"{path} states a {key} list fse cannot read: []")
        return [_string(item, key, path, pattern) for item in value]
    return _string(value, key, path, pattern)


def _string(value: Any, key: str, path: Path, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str):
        what = "that is no string" if key == "license" else "fse cannot read"
        raise CorruptArtifactError(f"{path} states a {key} {what}: {value!r}")
    if not pattern.fullmatch(value):
        raise CorruptArtifactError(f"{path} states a {key} fse cannot read: {value!r}")
    return value
