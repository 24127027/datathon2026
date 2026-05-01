from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_PROJECT_NAME = "datathon2026"


def prepare_project_root(
	project_name: str = DEFAULT_PROJECT_NAME,
) -> Path:
	"""Resolve the project root for local environments.

	Search common candidate paths and return the first valid project root.
	"""
	cwd = Path.cwd().parent.resolve()
	candidates = [cwd, cwd / project_name, cwd.parent / project_name]

	for candidate in candidates:
		if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
			return candidate

	return cwd


def bootstrap_project_paths(
	project_name: str = DEFAULT_PROJECT_NAME,
	verbose: bool = True,
) -> Path:
	"""Prepare project root and add root/src folders to sys.path.

	Returns:
		The resolved project root path.
	"""
	project_root = prepare_project_root(
		project_name=project_name,
	)
	src_dir = project_root / "src"

	for candidate in (project_root, src_dir):
		candidate_str = str(candidate)
		if candidate_str not in sys.path:
			sys.path.insert(0, candidate_str)

	if verbose:
		print(f"Project root: {project_root}")
		print(f"Source path added: {src_dir}")

	return project_root

