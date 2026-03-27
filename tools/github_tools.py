"""
tools/github_tools.py — GitHub repo management tools for the Dev agent.

All operations use PyGitHub to create repos, manage files, and create PRs.
"""

from __future__ import annotations

import base64
import json
import logging

from langchain_core.tools import tool
from github import Github, GithubException

from config.settings import github_cfg
from tools.retry import retry

logger = logging.getLogger(__name__)


_client: Github | None = None


def _get_client() -> Github:
    """Return a cached GitHub client instance."""
    global _client
    if _client is None:
        _client = Github(github_cfg.token)
    return _client


@retry(max_retries=3, base_delay=1.0, retryable_exceptions=(GithubException,))
def _github_call(fn, *args, **kwargs):
    """Call a GitHub API function with retry."""
    return fn(*args, **kwargs)


@tool
def github_create_repo(name: str, description: str = "") -> str:
    """Create a new private GitHub repository.
    Args: name (repo name, use kebab-case), description (optional).
    Returns: JSON with repo_url, clone_url, repo_name."""
    try:
        g = _get_client()
        user = g.get_user()
        repo = _github_call(
            user.create_repo,
            name=name,
            description=description,
            private=github_cfg.default_private,
            auto_init=True,
        )
        logger.info(f"GitHub repo created: {repo.full_name}")
        return json.dumps({
            "repo_url": repo.html_url,
            "clone_url": repo.clone_url,
            "repo_name": repo.full_name,
        })
    except GithubException as e:
        logger.error(f"GitHub create repo error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


@tool
def github_create_file(repo_name: str, file_path: str, content: str,
                       commit_message: str = "Add file") -> str:
    """Create or update a file in a GitHub repo.
    Args: repo_name (owner/repo format), file_path (e.g. 'src/main.py'),
    content (file content as string), commit_message."""
    try:
        g = _get_client()
        repo = g.get_repo(repo_name)
        try:
            existing = repo.get_contents(file_path)
            repo.update_file(file_path, commit_message, content, existing.sha)
            action = "updated"
        except GithubException:
            repo.create_file(file_path, commit_message, content)
            action = "created"
        logger.info(f"GitHub file {action}: {file_path} in {repo_name}")
        return json.dumps({"action": action, "file": file_path, "repo": repo_name})
    except GithubException as e:
        logger.error(f"GitHub create file error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


@tool
def github_read_file(repo_name: str, file_path: str) -> str:
    """Read a file from a GitHub repo.
    Args: repo_name (owner/repo format), file_path.
    Returns: file content as string."""
    try:
        g = _get_client()
        repo = g.get_repo(repo_name)
        content = repo.get_contents(file_path)
        return base64.b64decode(content.content).decode("utf-8")
    except GithubException as e:
        logger.error(f"GitHub read file error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


@tool
def github_create_pull_request(repo_name: str, title: str, body: str,
                               head_branch: str, base_branch: str = "main") -> str:
    """Create a pull request in a GitHub repo.
    Args: repo_name (owner/repo), title, body (PR description), head_branch (source), base_branch (target, default 'main').
    Returns: JSON with pr_url and pr_number."""
    try:
        g = _get_client()
        repo = g.get_repo(repo_name)
        pr = repo.create_pull(title=title, body=body, head=head_branch, base=base_branch)
        logger.info(f"GitHub PR created: {pr.html_url}")
        return json.dumps({"pr_url": pr.html_url, "pr_number": pr.number})
    except GithubException as e:
        logger.error(f"GitHub create PR error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


@tool
def github_list_files(repo_name: str, path: str = "") -> str:
    """List files in a GitHub repo directory.
    Args: repo_name (owner/repo), path (directory path, empty for root).
    Returns: JSON list of files with name, path, type."""
    try:
        g = _get_client()
        repo = g.get_repo(repo_name)
        contents = repo.get_contents(path or "")
        files = [{"name": f.name, "path": f.path, "type": f.type} for f in contents]
        return json.dumps(files)
    except GithubException as e:
        logger.error(f"GitHub list files error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


GITHUB_TOOLS = [github_create_repo, github_create_file, github_read_file,
                github_create_pull_request, github_list_files]
