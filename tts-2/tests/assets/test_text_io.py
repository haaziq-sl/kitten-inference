"""Every text-mode file access in tools/ names its encoding.

Python on Windows otherwise reads and writes text in the ANSI code page and writes CRLF line endings, which broke
model preparation there (UnicodeDecodeError on the voices JSON) and would change prepared files.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _text_mode(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "mode"), None)
    if mode is None:
        return True  # open(path) reads text
    return not (isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value)


def test_tools_text_io_names_its_encoding():
    bad = []
    for path in sorted((ROOT / "tools").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            keywords = {k.arg for k in node.keywords}
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text") and "encoding" not in keywords:
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno} {func.attr}() without encoding=")
            if isinstance(func, ast.Name) and func.id == "open" and _text_mode(node) and "encoding" not in keywords:
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno} text-mode open() without encoding=")
    assert not bad, "\n".join(bad)
