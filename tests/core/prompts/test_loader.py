"""Tests for prompt loader helpers."""

from pathlib import Path

from elspeth.core.prompts.loader import load_template, load_template_pair


def test_load_template_reads_file(tmp_path: Path):
    path = tmp_path / "template.jinja"
    path.write_text("Hello {{ name }}", encoding="utf-8")

    tmpl = load_template(path, name="greeting", defaults={"name": "world"})

    assert tmpl.name == "greeting"
    assert tmpl.render() == "Hello world"


def test_load_template_pair(tmp_path: Path):
    system_path = tmp_path / "system.md"
    user_path = tmp_path / "user.md"
    system_path.write_text("System here", encoding="utf-8")
    user_path.write_text("User: {{ value }}", encoding="utf-8")

    system, user = load_template_pair(system_path, user_path, defaults={"value": "x"})

    assert system.render() == "System here"
    assert user.render() == "User: x"
