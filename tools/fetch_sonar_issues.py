#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SONARCLOUD_ISSUES_API = "https://sonarcloud.io/api/issues/search"


def _issues_search_url(
    *,
    component_key: str,
    branch: str,
    pull_request: str,
    only_open: bool,
    page: int,
    page_size: int,
) -> str:
    params: dict[str, Any] = {
        "componentKeys": str(component_key),
        "p": int(page),
        "ps": int(page_size),
    }
    branch_s = str(branch).strip()
    pr_s = str(pull_request).strip()
    if branch_s:
        params["branch"] = branch_s
    if pr_s:
        params["pullRequest"] = pr_s
    if bool(only_open):
        params["resolved"] = "false"
    return f"{SONARCLOUD_ISSUES_API}?{urllib.parse.urlencode(params)}"


def _fetch_json(url: str, *, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch SonarCloud issues for a project/branch.")
    parser.add_argument("--component-key", default="skygazer42_ForeSight")
    parser.add_argument("--branch", default="")
    parser.add_argument("--pull-request", default="")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--token-env", default="SONAR_TOKEN")
    parser.add_argument("--output", default="")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--only-open", action="store_true", default=True)
    args = parser.parse_args(argv)

    token_env = str(args.token_env).strip() or "SONAR_TOKEN"
    token = os.environ.get(token_env, "")
    if not token:
        raise RuntimeError(f"Missing required environment variable: {token_env}")

    issues: list[dict[str, Any]] = []
    paging: dict[str, Any] = {}
    for page in range(1, int(args.max_pages) + 1):
        url = _issues_search_url(
            component_key=str(args.component_key),
            branch=str(args.branch),
            pull_request=str(args.pull_request),
            only_open=bool(args.only_open),
            page=page,
            page_size=int(args.page_size),
        )
        payload = _fetch_json(url, token=token)
        paging = dict(payload.get("paging", {}))
        page_issues = list(payload.get("issues", []))
        issues.extend(page_issues)
        if len(page_issues) < int(args.page_size):
            break

    result = {"paging": paging, "issues": issues}
    if str(args.format) == "json":
        text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        lines = [f"issues={len(issues)}"]
        for issue in issues:
            lines.append(
                " | ".join(
                    [
                        str(issue.get("severity", "")),
                        str(issue.get("type", "")),
                        str(issue.get("component", "")),
                        str(issue.get("message", "")),
                    ]
                )
            )
        text = "\n".join(lines)

    output = str(args.output).strip()
    if output:
        Path(output).write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    else:
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
