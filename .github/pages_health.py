#!/usr/bin/env python3
"""Check the health of the GitHub Pages site that `gh-pages` publishes.

The check fails when

  (a) the latest completed `pages-build-deployment` run failed, or
  (b) the expanded size of the `gh-pages` tree passes the limit (8 GiB by default).

GitHub Pages refuses a site above 10 GiB, and it expands a symbolic link to a directory into a
full copy of that directory. Documenter writes one such link per version alias (`stable`, `v0.32`,
`v0.1.0`, ...), so the size the deployment uploads is larger than the sum of the blobs. This check
counts every link as a copy of its target, the way the deployment does.

It reads the tree through the GitHub API, which reports the size of every blob, so it downloads
no blob except the few bytes of each link. It needs `gh` on the PATH and a token in `GH_TOKEN`.

    python3 .github/pages_health.py [owner/repo]

Environment: `PAGES_REPO` (default `dcelisgarza/PortfolioOptimisers.jl`), `PAGES_BRANCH`
(default `gh-pages`) and `PAGES_LIMIT_GIB` (default `8`).
"""

import base64
import json
import os
import posixpath
import subprocess
import sys

GIB = 1024**3
REPO = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAGES_REPO", "dcelisgarza/PortfolioOptimisers.jl")
BRANCH = os.environ.get("PAGES_BRANCH", "gh-pages")
LIMIT_GIB = float(os.environ.get("PAGES_LIMIT_GIB", "8"))
DEPLOY_WORKFLOW = "pages-build-deployment"
SHOWN = 15  # rows of the top-level table


def gh_json(args):
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def api(path):
    return gh_json(["api", path])


def walk(sha, prefix, out):
    """Collect every non-tree entry under the tree `sha` as `path -> (mode, size, sha)`.

    A recursive listing stops at 100 000 entries or 7 MB and sets `truncated`. A truncated tree is
    listed one level down instead, and each subtree is walked on its own, so no entry is missed.
    """
    rec = api(f"repos/{REPO}/git/trees/{sha}?recursive=1")
    if not rec["truncated"]:
        for e in rec["tree"]:
            if e["type"] != "tree":
                out[prefix + e["path"]] = (e["mode"], e.get("size", 0), e["sha"])
        return
    flat = api(f"repos/{REPO}/git/trees/{sha}")
    if flat["truncated"]:
        sys.exit(f"::error::The tree {prefix or '/'} has too many entries to list one level deep.")
    for e in flat["tree"]:
        if e["type"] == "tree":
            walk(e["sha"], prefix + e["path"] + "/", out)
        else:
            out[prefix + e["path"]] = (e["mode"], e.get("size", 0), e["sha"])


def link_target(sha, cache={}):
    if sha not in cache:
        blob = api(f"repos/{REPO}/git/blobs/{sha}")
        cache[sha] = base64.b64decode(blob["content"]).decode().strip()
    return cache[sha]


def expanded_sizes(entries):
    """Return the size of every path once each symbolic link is expanded into a copy of its target.

    The returned function maps a path (a file, a directory, or `""` for the root) to its expanded
    size in bytes.
    """
    plain = {}  # directory -> bytes of the regular files under it
    links = {}  # link path -> resolved target path
    for path, (mode, size, sha) in entries.items():
        if mode == "120000":
            links[path] = posixpath.normpath(posixpath.join(posixpath.dirname(path), link_target(sha)))
            continue
        if mode == "160000":  # a submodule has no content on the site
            continue
        plain[path] = size
        parts = path.split("/")
        for i in range(len(parts)):
            d = "/".join(parts[:i])
            plain[d] = plain.get(d, 0) + size

    def size_of(path, seen=()):
        if path in seen:
            sys.exit(f"::error::The symbolic links of {BRANCH} form a cycle through {path}.")
        if path in links:
            return size_of(links[path], seen + (path,))
        total = plain.get(path, 0)
        under = "" if path == "" else path + "/"
        for link in links:
            if link.startswith(under):
                total += size_of(links[link], seen + (path,))
        return total

    return size_of, links


def latest_deploy():
    runs = gh_json(["run", "list", "--repo", REPO, "--workflow", DEPLOY_WORKFLOW, "--limit", "20",
                    "--json", "status,conclusion,createdAt,url"])
    done = [r for r in runs if r["status"] == "completed"]
    return (done[0] if done else None), runs


def fmt(n):
    return f"{n / GIB:.2f} GiB"


def main():
    failures = []
    report = [f"## Pages health of `{REPO}`", ""]

    # (a) The latest completed deployment.
    run, runs = latest_deploy()
    if run is None:
        report.append(f"No completed `{DEPLOY_WORKFLOW}` run among the last {len(runs)}.")
    else:
        failed = 0
        for r in runs:
            if r["status"] != "completed":
                continue
            if r["conclusion"] == "success":
                break
            failed += 1
        report.append(f"Latest completed `{DEPLOY_WORKFLOW}`: **{run['conclusion']}** at {run['createdAt']} ({run['url']}).")
        if run["conclusion"] != "success":
            failures.append(
                f"The latest `{DEPLOY_WORKFLOW}` run ended `{run['conclusion']}` ({run['url']}); "
                f"{failed} completed run(s) in a row have not succeeded. The site is not being updated."
            )

    # (b) The expanded size of the tree.
    sha = api(f"repos/{REPO}/branches/{BRANCH}")["commit"]["sha"]
    root = api(f"repos/{REPO}/git/commits/{sha}")["tree"]["sha"]
    entries = {}
    walk(root, "", entries)
    size_of, links = expanded_sizes(entries)
    total = size_of("")
    blobs = sum(s for p, (m, s, _) in entries.items() if m not in ("120000", "160000"))
    top = sorted({p.split("/")[0] for p in entries})
    previews = sorted({"/".join(p.split("/")[:2]) for p in entries if p.startswith("previews/") and p.count("/") >= 2})
    previews_size = sum(size_of(p) for p in previews)

    report += [
        "",
        f"`{BRANCH}` at `{sha[:10]}`: {len(entries)} files, {len(links)} symbolic links.",
        "",
        "| | size |",
        "| --- | ---: |",
        f"| blobs, links unexpanded | {fmt(blobs)} |",
        f"| **expanded (what Pages uploads)** | **{fmt(total)}** |",
        f"| of which the {len(previews)} pull-request previews | {fmt(previews_size)} |",
        f"| of which the {len(links)} expanded links | {fmt(total - blobs)} |",
        f"| limit of this check | {fmt(LIMIT_GIB * GIB)} |",
        "| hard limit of GitHub Pages | 10.00 GiB |",
        "",
        f"The {SHOWN} largest of the {len(top)} top-level entries:",
        "",
        "| top-level entry | expanded size |",
        "| --- | ---: |",
    ]
    for name, n in sorted(((t, size_of(t)) for t in top), key=lambda x: -x[1])[:SHOWN]:
        arrow = f" -> `{links[name]}`" if name in links else ""
        report.append(f"| `{name}`{arrow} | {fmt(n)} |")
    if total > LIMIT_GIB * GIB:
        failures.append(
            f"The expanded size of `{BRANCH}` is {total / GIB:.2f} GiB, above the limit of {LIMIT_GIB:g} GiB "
            f"(GitHub Pages refuses a site above 10 GiB). Previews hold {previews_size / GIB:.2f} GiB and "
            f"the version links {(total - blobs) / GIB:.2f} GiB. Run the `Doc preview cleanup` workflow, "
            "or cap `deploydocs(versions = ...)` in `docs/make.jl`."
        )

    report.append("")
    report += [f"**FAIL**: {f}" for f in failures] or ["**OK**"]
    text = "\n".join(report)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(text + "\n")
    for f in failures:
        print(f"::error title=Pages health::{f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
