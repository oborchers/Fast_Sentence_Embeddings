"""Extract documentation references without interpreting examples as links."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser

from markdown_it import MarkdownIt


@dataclass(frozen=True)
class Reference:
    target: str
    line: int
    cell: int | None = None
    inline: bool = False
    attachments: tuple[str, ...] = ()


class Attributes(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.references = []

    def handle_starttag(self, _tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value is not None:
                self.references.append((value, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def html_references(text, line, cell, attachments):
    parser = Attributes()
    parser.feed(text)
    return [
        Reference(target, line + offset - 1, cell, attachments=attachments)
        for target, offset in parser.references
    ]


def inline_references(token, line, cell, attachments):
    if token.type == "html_inline":
        return html_references(token.content, line, cell, attachments)
    if token.type in {"link_open", "image"}:
        target = token.attrGet("href" if token.type == "link_open" else "src")
        return [Reference(target, line, cell, attachments=attachments)]
    if token.type == "code_inline":
        return [Reference(token.content, line, cell, True)]
    return []


def inline_tokens_references(children, line, cell, attachments):
    found = []
    depth = 0
    for child in children:
        if child.type == "link_close":
            depth -= 1
        if not (depth and child.type == "code_inline"):
            found.extend(inline_references(child, line, cell, attachments))
        if child.type == "link_open":
            depth += 1
    return found


def markdown_references(text, cell=None, attachments=()):
    parser = MarkdownIt("commonmark").enable("table")
    found = []
    for token in parser.parse(text):
        line = token.map[0] + 1 if token.map else 1
        if token.type == "html_block":
            found.extend(html_references(token.content, line, cell, attachments))
        found.extend(inline_tokens_references(token.children or (), line, cell, attachments))
    return found


def cell_markdown(cell, number):
    if not isinstance(cell, dict):
        raise ValueError(f"cell {number} must be an object")
    if cell.get("cell_type") != "markdown":
        return []
    source = cell.get("source")
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        source = "".join(source)
    if not isinstance(source, str):
        raise ValueError(f"cell {number} source must be a string or list of strings")
    attachments = cell.get("attachments", {})
    if not isinstance(attachments, dict):
        raise ValueError(f"cell {number} attachments must be an object")
    return markdown_references(source, number, tuple(attachments))


def notebook_references(text):
    notebook = json.loads(text)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("notebook must contain a cells list")
    found = []
    for number, cell in enumerate(notebook["cells"], 1):
        found.extend(cell_markdown(cell, number))
    return found
