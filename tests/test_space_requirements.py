import ast
import re
from pathlib import Path


def test_space_requirements_are_standalone_and_include_core_dependencies():
    root = Path(__file__).resolve().parents[1]
    requirements = [line.strip() for line in (root / "requirements.txt").read_text(
        encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any(line.startswith((".", "-e")) for line in requirements)

    project = (root / "pyproject.toml").read_text(encoding="utf-8").split("[project]", 1)[1]
    project = project.split("\n[", 1)[0]
    dependencies = ast.literal_eval(re.search(r"dependencies\s*=\s*(\[.*?\])", project, re.DOTALL)[1])

    def name(spec):
        return re.match(r"[\w.-]+", spec)[0].lower().replace("_", "-")

    assert {name(spec) for spec in dependencies} <= {name(spec) for spec in requirements}
