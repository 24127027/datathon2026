from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_BASE_PATH = Path("/kaggle/working")
DEFAULT_PROJECT_NAME = "datathon2026"


def _is_kaggle_runtime() -> bool:
	"""Return True when running inside a Kaggle notebook runtime."""
	return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def _get_github_token() -> str | None:
	"""Read GITHUB_TOKEN from Kaggle Secrets if available."""
	if not _is_kaggle_runtime():
		return None

	try:
		from kaggle_secrets import UserSecretsClient

		user_secrets = UserSecretsClient()
		token = user_secrets.get_secret("GITHUB_TOKEN")
		return token or None
	except Exception:
		return None


def _build_repo_url(repo_url: str) -> str:
	"""Embed a GitHub token into an HTTPS URL when available in Kaggle."""
	token = _get_github_token()
	if not token:
		return repo_url

	prefix = "https://"
	if repo_url.startswith(prefix):
		return repo_url.replace(prefix, f"https://x-access-token:{token}@", 1)
	return repo_url


def _run_git(args: list[str], cwd: Path) -> None:
	subprocess.run(args, cwd=cwd, check=True)


def prepare_project_root(
	base_path: Path = DEFAULT_BASE_PATH,
	project_name: str = DEFAULT_PROJECT_NAME,
	repo_url: str | None = None,
) -> Path:
	"""Resolve or prepare project root for local and Kaggle environments.

	In Kaggle:
	- If the project folder does not exist and repo_url is provided, clone it.
	- If the project folder exists, attempt a git pull.

	Locally:
	- Search common candidate paths and return the first valid project root.
	"""
	if _is_kaggle_runtime():
		project_root = base_path / project_name
		has_token = bool(_get_github_token())

		if not project_root.exists():
			if not repo_url:
				raise RuntimeError(
					"Kaggle runtime detected but project folder is missing and no repo_url was provided."
				)

			clone_url = _build_repo_url(repo_url)
			try:
				_run_git(["git", "clone", clone_url, str(project_root)], cwd=base_path)
				print("Repository cloned successfully.")
			except subprocess.CalledProcessError as exc:
				if not has_token:
					raise RuntimeError(
						"Failed to clone repository on Kaggle. Add GITHUB_TOKEN in Kaggle Secrets and rerun."
					) from exc
				raise
		else:
			try:
				_run_git(["git", "pull"], cwd=project_root)
				print("Repository updated successfully.")
			except subprocess.CalledProcessError as exc:
				if not has_token:
					raise RuntimeError(
						"Failed to pull repository on Kaggle. Add GITHUB_TOKEN in Kaggle Secrets and rerun."
					) from exc
				raise

		return project_root

	cwd = Path.cwd().parent.resolve()
	candidates = [cwd, cwd / project_name, cwd.parent / project_name]

	for candidate in candidates:
		if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
			return candidate

	return cwd


def bootstrap_project_paths(
	repo_url: str | None = None,
	base_path: Path = DEFAULT_BASE_PATH,
	project_name: str = DEFAULT_PROJECT_NAME,
	verbose: bool = True,
) -> Path:
	"""Prepare project root and add root/src folders to sys.path.

	Returns:
		The resolved project root path.
	"""
	project_root = prepare_project_root(
		base_path=base_path,
		project_name=project_name,
		repo_url=repo_url,
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

